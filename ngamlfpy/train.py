"""
Training/prediction related classes.

Main classes are:

* **Trainer**
* **AzureTrainer** (subclass of Trainer, provides additional functionality to log to Azure etc)

Currently available learning algorithms within the Trainer/AzureTrainer classes are:

* scikit_neural_network_regressor
* scikit_isolation_forest
* enhanced_isolation_forest (NGA proprietary algorithm)

The relevant training algorithm to use for a particular ml service is specified in the learningAlgorithm attribute of the ml service (see hrx mlconfig api ml-services endpoint)


"""

from ngamlfpy.pipeline import FileFinder

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib

import numpy as np
import matplotlib.pyplot as plt
from ngamlfpy.hrxmlconfig import MLModelConfig
from ngamlfpy.algorithms import EnhancedIsolationForest
import os

from azureml.core import Run
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.model import Model


import datetime
import time

class SKDecisionForestAnalyser:
    """
    Used to analyse standard scikit learn decision tree classifiers  to determine features most used for each sample.

    Main entry point: determine_important_features()

    """
    def __init__(self,trained_model,feat_name_list = []):
        """
        :param trained_model: Needs to be a model based on scikit learn DecsionTreeClassifer
        :param feat_name_list: Feature names
        """
        self.trained_model = trained_model
        self.feat_name_list = feat_name_list

    def _get_indices(self, est, samples):
        """ Tree based learning algorithms only """

        indices_ar = []
        dec_paths = est.decision_path(samples)
        for i in range(0, len(samples)):
            indices_ar.append(dec_paths.indices[dec_paths.indptr[i]:dec_paths.indptr[i + 1]])
        return np.array(indices_ar)


    def _is_leaf(self, tr, ind):
        """ Tree based learning algorithms only """

        if ((tr.children_left[ind] == -1) and (tr.children_right[ind] == -1)):
            return True

        return False

    def _get_parent_ind_all_inds(self, tr):
        """ Tree based learning algorithms only """

        ch_left = tr.children_left
        ch_right = tr.children_right

        parent_dict = {}

        for p_num, child_num in enumerate(ch_left):
            if (child_num == -1):
                pass
            else:
                parent_dict[child_num] = p_num

        for p_num, child_num in enumerate(ch_right):
            if (child_num == -1):
                pass
            else:
                parent_dict[child_num] = p_num

        return parent_dict

    def _get_parent_ind(self, tr, ind, parent_dict=None):
        """ Tree based learning algorithms only """

        if (parent_dict is None):
            ch_left = tr.children_left
            ch_right = tr.children_right
            for i, entry in enumerate(ch_left):
                if entry == ind:
                    return i
            for i, entry in enumerate(ch_right):
                if entry == ind:
                    return i
        else:
            return parent_dict[ind]

    def _get_feature_used(self, tr, ind, parent_dict=None):
        """ Tree based learning algorithms only """

        f_used = tr.feature[ind]
        if (f_used == -2):
            p_id = self._get_parent_ind(tr, ind, parent_dict)
            return tr.feature[p_id]
        else:
            return f_used


    def _process_est_tree_sample(self, sample_num, indices, tr, feature_used_list_all_trees, last_feature_used_list_all_trees, parent_dict):
        """ Tree based learning algorithms only """

        operations_this_sample = 0
        feature_used_dict = feature_used_list_all_trees[sample_num]
        last_feature_used_dict = last_feature_used_list_all_trees[sample_num]
        for ind in indices:
            operations_this_sample +=1
            if (self._is_leaf(tr, ind)):
                last_feat = self._get_feature_used(tr, ind, parent_dict)
                if (last_feat in last_feature_used_dict):
                    last_feature_used_dict[last_feat] += 1
                else:
                    last_feature_used_dict[last_feat] = 1

            f_used = self._get_feature_used(tr, ind, parent_dict)
            if f_used in feature_used_dict:
                feature_used_dict[f_used] += 1
            else:
                feature_used_dict[f_used] = 1

        return operations_this_sample

    def _process_est_tree(self, tree_num, est, samples, feature_used_list_all_trees, last_feature_used_list_all_trees,
                          show_progress_after_each=-1, show_diagnostics_flag=False):
        """ Tree based learning algorithms only """

        startTree = int(time.time() * 1000)
        if show_progress_after_each != -1:
            if tree_num % show_progress_after_each == 0:
                print('Processing tree num: ', tree_num + 1, ' / ', len(self.trained_model.estimators_))
        tr = est.tree_
        ind_sets = self._get_indices(est, samples)
        parent_dict = self._get_parent_ind_all_inds(tr)
        tot_operations = 0

        for sample_num, indices in enumerate(ind_sets):
            tot_operations += self._process_est_tree_sample(sample_num, indices, tr, feature_used_list_all_trees, last_feature_used_list_all_trees, parent_dict)

        end_tree = int(time.time() * 1000)

        if ((show_progress_after_each != -1) and (show_diagnostics_flag)):
            if (tree_num % show_progress_after_each == 0):
                print('Tree: %s took: %s ms. Tot ops performed: %s' % (tree_num + 1, end_tree - startTree, tot_operations))

    def _accum_features_used(self, df_X_scaled, show_progress_after_each=-1, show_diagnostics_flag=False, num_trees_to_check=None):
        """ Tree based learning algorithms only """
        # set num_trees_to_check to a specific number if wanting to restrict to a few trees for testing purposes

        start_accum = int(time.time() * 1000)
        i = 0

        samples = df_X_scaled.values
        print('\nDetermining important features per sample')
        feature_used_list_all_trees = [{} for i in range(0, len(samples))]
        last_feature_used_list_all_trees = [{} for i in range(0, len(samples))]
        for tree_num, est in enumerate(self.trained_model.estimators_):
            if num_trees_to_check is None:
                pass
            else:
                if tree_num > num_trees_to_check:
                    break

            self._process_est_tree(tree_num, est, samples, feature_used_list_all_trees,
                                   last_feature_used_list_all_trees, show_progress_after_each=show_progress_after_each,
                                   show_diagnostics_flag=show_diagnostics_flag)

        end_accum = int(time.time() * 1000)
        if (show_diagnostics_flag):
            print('Overall time in accum: ', end_accum - start_accum)

        return feature_used_list_all_trees, last_feature_used_list_all_trees

    def _top_features_used(self, fu_array, feat_name_list, num_top_features=3):
        """ Tree based learning algorithms only """
        sorted_ar = []

        feat_name_list_ext = feat_name_list[:]
        feat_name_list_ext.append('.') # Use when no top feature is available for a sample

        min_top_feats = 9999

        for entry in fu_array:
            tmp = sorted(entry, key=entry.get, reverse=True)
            if len(tmp) < min_top_feats:
                min_top_feats = len(tmp)
            if len(tmp) < num_top_features:
                tmp.extend([len(feat_name_list_ext) - 1 for n in range(len(tmp), num_top_features)])
            sorted_ar.append(tmp[:num_top_features])

        print('Min top features: ',min_top_feats)
        named_ar = []
        for entry in sorted_ar:
            new_entry = [feat_name_list_ext[x] for x in entry]
            named_ar.append(new_entry)
        return np.array(named_ar)


    def determine_important_features(self, df_X_scaled, df, num_top_features, show_progress_after_each=20,
                                     show_diagnostics_flag=False):
        """

        :param df_X_scaled:  scaled df with features only
        :param df:  master df (can include other id columns)
        :param num_top_features: How many most important features to return per sample
        :param show_progress_after_each:
        :param show_diagnostics_flag:
        :return: df with most important feature columns added
        """

        feat_name_list = self.feat_name_list

        fu = self._accum_features_used(df_X_scaled, show_progress_after_each, show_diagnostics_flag, num_trees_to_check=None)

        if (show_diagnostics_flag):
            print(' ')
            print('First 5 samples total feature usage: ')
            for entry in fu[0][:5]:
                print(entry)
            print(' ')
            print('First 5 samples last feature usage: ')
            for entry in fu[1][:5]:
                print(entry)

        # Check top features used per example over all tree branches
        fu_overall = self._top_features_used(fu[0], feat_name_list, num_top_features)
        if (show_diagnostics_flag):
            print(' ')
            print('First 5 samples top n overall features used: ')
            for entry in fu_overall[:5]:
                print(entry)

        # Check top features used  per example as last branch only of tree path
        fu_last = self._top_features_used(fu[1], feat_name_list, num_top_features)
        if (show_diagnostics_flag):
            print(' ')
            print('First 5 samples top n last features used: ')
            for entry in fu_last[:5]:
                print(entry)

        df_with_feature_importances = df.copy()

        for i in range(0, num_top_features):
            df_with_feature_importances['F' + str(i + 1)] = fu_overall[:, i]

        for i in range(0, num_top_features):
            df_with_feature_importances['LF' + str(i + 1)] = fu_last[:, i]

        return df_with_feature_importances


class Trainer:
    """
     |Base class used to train hrx ml py framework models.
     |Note: Mostly the subclass AzureTrainer will be used, which includes functionality to log to Azure etc
     """

    trainer_code_version = '0.1d'

    def __init__(self,ml_service,model_name,hyper_params=None,in_trained_model=None,in_scaler_X=None,in_scaler_Y=None,base_folder=None, model_version=None, clip_training_set=None, clip_test_set=None):
        """
        :param ml_service:  ML Service, eg TWV or PAD
        :param model_name:  eg T001
        :param hyper_params: Optional. If not supplied, uses defaults
        :param in_trained_model: Optional already trained model. If supplied, bypasses loading of training set source files - since  no training is needed
        :param in_scaler_X: Optional: for already trained models, scaler objects supplied for normalisation
        :param in_scaler_Y: Optional: for already trained models, scaler objects supplied for normalisation
        :param base_folder:  Optional (defaults to  ./data if not supplied)
        :param model_version: Optional model version number to train. If not supplied, trains latest model version

        Note: static factory method usually used to help construct Trainer object: trainer_from_args
        """

        self.model_name = model_name
        self.ml_service = ml_service

        self.ml_service_details = MLModelConfig.get_ml_service_details(self.ml_service)

        if hyper_params is None:
            self.hyper_params = {'run_num': 1, 'learning_rate_init': 0.001, 'alpha': 1.0,
                            'hidden_layer_sizes': [100], 'num_iters': 500, 'max_iter': 50000,
                            'batch_size': 200, 'iso_num_estimators':200, 'iso_max_samples':'auto','iso_max_features':'1.0','clip_training_set':-1, 'clip_test_set':-1}
        else:
            self.hyper_params = hyper_params

        if clip_training_set is None or clip_training_set == -1:
            pass
        else:
            self.hyper_params['clip_training_set'] = clip_training_set

        if clip_test_set is None or clip_test_set == -1:
            pass
        else:
            self.hyper_params['clip_test_set'] = clip_test_set

        # self.trained_model = None
        #
        # self.load_aready_trained_model = load_already_trained_model
        #
        # if self.load_aready_trained_model:
        #    self.trained_model = self.get_model(run_type='run_local_only')
        self.trained_model = in_trained_model
        self.scaler_X = in_scaler_X
        self.scaler_Y = in_scaler_Y

        #for when trained model is already supplied for prediction mode

        self.model_config = MLModelConfig.get_model_config_from_web_service_for_model(ml_service, model_name)

        if model_version is None:
            self.model_version_to_train = self.model_config.get_model_version()
        else:
            self.model_version_to_train = model_version

        self.finder = FileFinder(self.ml_service, use_model_name_in=True, use_model_name_out=True,
                            model_name=self.model_name, model_version=self.model_version_to_train,
                            relative_input_folder='train', relative_output_folder='train',base_folder=base_folder)



        if in_trained_model is None:
            self.read_all_training_files() # Only read if training needs to be done

    @staticmethod
    def trainer_from_args(args):
        """
        Static method used to help construct Trainer object, based on command line args provided.

        args include:
         - data_folder
         - model
         - ml_service
         - model_version (If not specified, uses latest version)
         - hp_lr_init (Hyperparameter - initial learning rate, eg 0.001)
         - hp_reg (Hyperparameter - regularisation alpha, eg 0.0001)
         - hp_hidden_layers(Hyperparameter - neural network hidden layers, eg [100])
         - hp_iters (Hyperparameter - number iterations, eg 1000)
         - hp_iso_num_estimators ((Hyperparameter - isolation forest num estimators, eg 200
         - hp_iso_max_samples (Hyperparameter - isolation forest max samples, eg '0.9')
         - hp_iso_max_features (Hyperparameter - isolation  forest max features, eg '1.0')
         - clip_training_set (Optional. Number of training examples to restrict to - uses all available via train/test split if not supplied)
         - clip_test_set (Optional. Number of test examples to restrict to - uses all available via train/test split if not supplied)
         - local_run_log_to_azure (default 'N'. Specifies whether to still log to Azure if it is a local run
         - local_log_experiment_name (default 'deep_neural_network'. Specifies what Azure experiment to log to if running locally with Azure logging on
         - local_log_ws_config_path (Optional - specifies for local runs the path which contained the Azure ws_config


        """

        print('hidden layers: ', args.hp_hidden_layers)

        data_folder = args.data_folder
        print('Data folder:', data_folder)

        model = args.model
        model_vers = args.model_version
        ml_service = args.ml_service
        clip_training_set = args.clip_training_set
        clip_test_set = args.clip_test_set


        if '.' in args.hp_iso_max_samples:
            iso_max_samples = float(args.hp_iso_max_samples) # fraction of samples to use
        else:
            try:
                iso_max_samples = int(args.hp_iso_max_samples) # number of samples to use
            except:
                iso_max_samples = args.hp_iso_max_samples # string, eg 'auto'

        if '.' in args.hp_iso_max_features:
            iso_max_features = float(args.hp_iso_max_features) # fraction of features to use
        else:
            iso_max_features = int(args.hp_iso_max_features) # number of features  to use


        hyper_params = {'run_num': 1, 'learning_rate_init': args.hp_lr_init, 'alpha': args.hp_reg,
                        'hidden_layer_sizes': args.hp_hidden_layers, 'num_iters': args.hp_iters, 'max_iter': 50000,
                        'iso_num_estimators' : args.hp_iso_num_estimators, 'iso_max_samples': iso_max_samples,
                        'iso_max_features': iso_max_features,
                        'batch_size': 200, 'clip_training_set':args.clip_training_set, 'clip_test_set': args.clip_test_set}

        return Trainer(ml_service,model,hyper_params,model_version=model_vers,clip_training_set=clip_training_set, clip_test_set=clip_test_set)


    def get_model(self,run_type):
        """ Deprecated """

        #global model, nn_model, scaler_X, scaler_Y, j_config_from_model

        if run_type == 'run_local_only':
            # period = 'MULTI'
            # model_file_name = 'data/train/' + model + '_' + target_vers + '/' + model + '_' + target_vers + '_dnn_model_run' + str(run_num) + '.pkl'
            # downloaded model name:
            model_path = 'data/' + self.ml_service + '/train/' + self.model_name + '/'
            model_name = self.ml_service + '_model_' + self.model_name
            model_file_name = model_path + model_name + '.pkl'  # 'twv_dnn_model_'+ model_code  + '.pkl'
            model_objects_list = joblib.load(model_file_name)
        else:
            model_name = 'twv_dnn_' + self.model_name
            model_path = Model.get_model_path(
                model_name)  # ('twv_dnn_' + model_code) # +  '_' + target_vers + '_run' + str(run_num))
            model_objects_list = joblib.load(model_path)

        trained_model = model_objects_list[0]
        #scaler_X = model_objects_list[1]
        #scaler_Y = model_objects_list[2]
        #j_config_from_model = model_objects_list[3]

        return trained_model

    def is_unsupervised(self):
        """ Check if ml service being trained is unsupervised """

        if 'mlType' in self.ml_service_details:
            if self.ml_service_details['mlType'] == 'unsupervised':
                return True
            else:
                return False

        return False

    def get_learning_algorithm(self):
        """ Get learning algorithm used by ml service """

        if 'learningAlgorithm' in self.ml_service_details:
            return self.ml_service_details['learningAlgorithm']

        return None


    def clip_training_set(self):
        """ Clip training set based on hyper param "clip_training_set" supplied in constructor """

        clip_max = self.hyper_params['clip_training_set']
        self.df_train_master = self.df_train_master[:clip_max]
        self.df_X_train = self.df_X_train[:clip_max]
        self.df_X_train_scaled = self.df_X_train_scaled[:clip_max]
        if self.is_unsupervised():
            pass
        else:
            self.df_Y_train = self.df_Y_train[:clip_max]
            self.df_Y_train_scaled = self.df_Y_train_scaled[:clip_max]


        print('df train master clipped: ' + str(self.df_train_master.shape))


    def clip_test_set(self):
        """ Clip test set based on hyper param "clip_test_set" supplied in constructor """

        clip_max = self.hyper_params['clip_test_set']
        self.df_test_master = self.df_test_master[:clip_max]
        self.df_X_test = self.df_X_test[:clip_max]
        self.df_X_test_scaled = self.df_X_test_scaled[:clip_max]
        if self.is_unsupervised():
            pass
        else:
            self.df_Y_test = self.df_Y_test[:clip_max]
            self.df_Y_test_scaled = self.df_Y_test_scaled[:clip_max]


        print('df test master clipped: ' + str(self.df_test_master.shape))


    def read_all_training_files(self):
        """ If no pre-trained model is supplied to constructor, reads all training set files related to model and version  being trained,
         in preparation for training """

        model = self.model_name

        X_train_name = os.path.join(self.finder.get_input_folder(), model + '_X_train.csv')
        X_test_name = os.path.join(self.finder.get_input_folder(), model + '_X_test.csv')
        Y_train_name = os.path.join(self.finder.get_input_folder(), model + '_Y_train.csv')
        Y_test_name = os.path.join(self.finder.get_input_folder(), model + '_Y_test.csv')

        self.df_X_train = pd.read_csv(X_train_name)
        self.df_X_test = pd.read_csv(X_test_name)

        if self.is_unsupervised():
            self.df_Y_train = None
            self.df_Y_test = None
        else:
            self.df_Y_train = pd.read_csv(Y_train_name)
            self.df_Y_test = pd.read_csv(Y_test_name)

        X_train_scaled_name = os.path.join(self.finder.get_input_folder(), model + '_X_train_scaled.csv')
        X_test_scaled_name = os.path.join(self.finder.get_input_folder(), model + '_X_test_scaled.csv')
        Y_train_scaled_name = os.path.join(self.finder.get_input_folder(), model + '_Y_train_scaled.csv')
        Y_test_scaled_name = os.path.join(self.finder.get_input_folder(), model + '_Y_test_scaled.csv')

        self.df_X_train_scaled = pd.read_csv(X_train_scaled_name)
        self.df_X_test_scaled = pd.read_csv(X_test_scaled_name)
        if self.is_unsupervised():
            self.df_Y_train_scaled = None
            self.df_Y_test_scaled = None
        else:
            self.df_Y_train_scaled = pd.read_csv(Y_train_scaled_name)
            self.df_Y_test_scaled = pd.read_csv(Y_test_scaled_name)

        print('X train scaled: ' + str(self.df_X_train_scaled.shape))
        print('X test scaled: ' + str(self.df_X_test_scaled.shape))
        if self.is_unsupervised():
            pass
        else:
            print('Y train scaled: ' + str(self.df_Y_train_scaled.shape))
            print('Y test scaled: ' + str(self.df_Y_test_scaled.shape))

        df_train_master_name = os.path.join(self.finder.get_input_folder(), model + '_train_master.csv')
        df_test_master_name = os.path.join(self.finder.get_input_folder(), model + '_test_master.csv')

        self.df_train_master = pd.read_csv(df_train_master_name)
        self.df_test_master = pd.read_csv(df_test_master_name)

        print('df train master: ' + str(self.df_train_master.shape))
        print('df test master: ' + str(self.df_test_master.shape))
        self.df_train_master.head()

        scaler_X_name = os.path.join(self.finder.get_input_folder(), model + '_scaler_X_model.pkl')
        scaler_Y_name = os.path.join(self.finder.get_input_folder(), model + '_scaler_Y_model.pkl')

        self.scaler_X = joblib.load(scaler_X_name)

        if self.is_unsupervised():
            self.scaler_Y = None
        else:
            self.scaler_Y = joblib.load(scaler_Y_name)


        if  (self.hyper_params['clip_training_set'] is None)or (self.hyper_params['clip_training_set'] == -1):
            pass
        else:
            self.clip_training_set()

        if (self.hyper_params['clip_test_set'] is None) or (self.hyper_params['clip_test_set'] == -1):
            pass
        else:
            self.clip_test_set()


    def write_model(self):
        """Write trained model to disk as .pkl

        File name is in format: <ml_service>_model_<model name>_<model version to train>.pkl

        Objects written to .pkl file consist of a single Python list containing:
            - trained model
            - scaler_X used
            - scaler_Y used
            - JSON model config

        Note: .pkl file is written to special "outputs" directory for Azure use
        """

        model_file_name = self.ml_service + '_model_' + self.model_name + '_' + self.model_version_to_train  +'.pkl'
        model_full_file_name = os.path.join(self.finder.get_output_folder(),model_file_name)
        model_objects_list = [self.trained_model, self.scaler_X, self.scaler_Y, self.model_config.j_config]
        os.makedirs(self.finder.get_output_folder(), exist_ok=True)
        joblib.dump(value=model_objects_list, filename=model_full_file_name)



    def stats(self, epoch, costs):
        """
         Prints stats relating to a training epoch.

         For unsupervised models:
          - None

         For supervised models:
           - Train cost / Test cost
           - Train accuracy / Test accuracy

         Returns: train cost, test cost, train accuracy, test accuracy (all zero if unsupervised)
        """

        if self.is_unsupervised():
            avg_cost = 0
            avg_test_cost = 0
            costs.append([avg_cost, avg_test_cost])
            acc_trn = 0
            acc_tst = 0
        else:
            d_train_withp = self.add_prediction_column(self.df_train_master, self.df_X_train_scaled)
            avg_cost = d_train_withp.sum()['cost'] / self.df_train_master.shape[0]
            d_test_withp = self.add_prediction_column(self.df_test_master, self.df_X_test_scaled)
            avg_test_cost = d_test_withp.sum()['cost'] / self.df_test_master.shape[0]
            costs.append([avg_cost, avg_test_cost])
            acc_trn = self.accuracy(d_train_withp)
            acc_tst = self.accuracy(d_test_withp)
        str_format = "{0:.3f}"

        if self.is_unsupervised():
            print(' Epoch: ' + str(epoch) + ' Unsupervised train..')
        else:
            print(' Epoch: ' + str(epoch) + ' Score train / test: ' + str(
                str_format.format(self.trained_model.score(self.df_X_train_scaled, self.df_Y_train_scaled))) + ' / ' + str(
                str_format.format(self.trained_model.score(self.df_X_test_scaled, self.df_Y_test_scaled))) + ' Av cost trn/test: ' + str(
                str_format.format(avg_cost)) + ' / ' + str(str_format.format(avg_test_cost)) + ' Acc trn/test: ' + str(
                str_format.format(acc_trn)) + ' / ' + str(str_format.format(acc_tst)))
        return avg_cost, avg_test_cost, acc_trn, acc_tst


    def determine_important_features(self, df_X_scaled, df, num_top_features, show_progress_after_each=20,
                                     show_diagnostics_flag=False):
        """ Tree based learning algorithms only """

        feat_name_list = self.model_config.get_feature_field_names()

        if self.ml_service_details['learningAlgorithm'] == 'enhanced_isolation_forest':
            df_with_feature_importances = self.trained_model.determine_important_features(df_X_scaled, df, num_top_features,
                                                                                     show_progress_after_each=show_progress_after_each,
                                                                                     show_diagnostics_flag=show_diagnostics_flag)
        else:
            # Standard scikit learn algorithm has no feature importances method, so use our SKDecisionForestAnalyser
            tree_analyser = SKDecisionForestAnalyser(self.trained_model, feat_name_list)
            df_with_feature_importances = tree_analyser.determine_important_features(df_X_scaled, df, num_top_features, show_progress_after_each=show_progress_after_each, show_diagnostics_flag=show_diagnostics_flag)

        return df_with_feature_importances


    def add_score_column(self,df,df_X_scaled,sort_by_scores=False):
        """ For Isolation Forest learning algorithms: perform predictions and add anomaly score column to dataframe for each employee """
        # for Isolation Forest

        scores = self.trained_model.decision_function(df_X_scaled)

        df_with_scores = df.copy()
        df_with_scores['score'] = scores

        if sort_by_scores:
            df_with_scores = df_with_scores.sort_values(by=['score'])
            df_with_scores['rank'] = [i+1 for i in range(df_with_scores.shape[0])]


        return df_with_scores


    def add_prediction_column(self,df, df_X_scaled,pred_set=False):
        """ For supervised learning algorithms: perform predictions and add prediction column to dataframe for each employee """

        label_col = self.model_config.get_label_col()

        label_col_details = self.model_config.get_field_with_title(label_col)

        p = self.trained_model.predict(df_X_scaled)
        if self.scaler_Y is None:
            pass
        else:
            np_p = np.array(p)
            np_p = np_p.reshape([len(p), 1])
            p = self.scaler_Y.inverse_transform(np_p)
        df_with_pred = df.copy()
        df_with_pred['pred'] = p
        df_with_pred = df_with_pred.round({'pred': 2})

        if label_col_details is None:
            pass
        else:
            if label_col_details['data_type'] == 'F':
                df_with_pred[label_col] = df_with_pred[label_col].astype(float)
            elif label_col_details['data_type'] == 'N':
                df_with_pred[label_col] = df_with_pred[label_col].astype(int)

        if pred_set:
            pass
        else:
            if self.is_unsupervised():
                pass
            else:
                df_with_pred['diff'] = df_with_pred['pred'] - df_with_pred[label_col]
                df_with_pred['perc_diff'] = df_with_pred['diff'] / df_with_pred[label_col] * 100.0
                df_with_pred = df_with_pred.round({'perc_diff': 2})
                df_with_pred['cost'] = df_with_pred['diff'] * df_with_pred['diff']

        return df_with_pred

    def accuracy(self,df_with_predict, threshold=2.0):
        # threshold = maximum percentage difference for a prediction to count as correct
        num_correct = df_with_predict[abs(df_with_predict.perc_diff) <= threshold].count()['perc_diff']
        return num_correct / df_with_predict.shape[0] * 100

    def _initialise_trained_model(self):

        learning_algorithm = self.get_learning_algorithm()

        trained_model = None

        if learning_algorithm == 'scikit_neural_network_regressor':
           trained_model = self._create_nn_regressor()

        elif learning_algorithm == 'scikit_isolation_forest':
            trained_model = self._create_isolation_forest()

        elif learning_algorithm == 'enhanced_isolation_forest':
            trained_model = self._create_enhanced_isolation_forest()


        return trained_model

    def _train_model(self, report_progress):

        learning_algorithm = self.get_learning_algorithm()

        if learning_algorithm == 'scikit_neural_network_regressor':
            return self._train_nn_regressor(report_progress)

        elif (learning_algorithm == 'scikit_isolation_forest') or (learning_algorithm == 'enhanced_isolation_forest'):
            return self._train_isolation_forest(report_progress)

        return None

    def _train_nn_regressor(self, report_progress):

        curr_epoch = 0
        num_iters = self.hyper_params['num_iters']
        costs = []

        for i in range(curr_epoch, curr_epoch + num_iters):
            self.trained_model.partial_fit(self.df_X_train_scaled, self.df_Y_train_scaled.values.ravel())
            # nn.fit(df_X_train_scaled, df_Y_train_scaled.values.ravel())
            # stats(run_num, i, nn, df_train, df_X_train_scaled, df_test, df_X_test_scaled, scaler_Y, costs, j_config)
            if i % 100 == 0:
                if report_progress:
                    #              print('i: ' + str(i))
                    self.stats(i, costs)

            # nn.fit(df_X_train_scaled, df_Y_train_scaled.values.ravel())
            # i = 0

        avg_cost, avg_test_cost, acc_trn, acc_tst = self.stats(i, costs)
        return avg_cost, avg_test_cost, acc_trn, acc_tst, i


    def _train_isolation_forest(self, report_progress):

        self.trained_model.fit(self.df_X_train_scaled)

        i = 1
        costs = []
        avg_cost, avg_test_cost, acc_trn, acc_tst = self.stats(i, costs)
        return avg_cost, avg_test_cost, acc_trn, acc_tst, i

    def _create_nn_regressor(self):

        learning_rate_init = self.hyper_params['learning_rate_init']
        alpha = self.hyper_params['alpha']
        hidden_layer_sizes = self.hyper_params['hidden_layer_sizes']
        max_iter = self.hyper_params['max_iter']
        batch_size = self.hyper_params['batch_size']

        trained_model = MLPRegressor(random_state=4242, alpha=alpha, max_iter=max_iter,
                                          hidden_layer_sizes=hidden_layer_sizes,
                                          batch_size=batch_size,
                                          learning_rate_init=learning_rate_init)  # , warm_start=True)  # (max_iter=200)

        return trained_model

    def _create_isolation_forest(self):

        iso_num_estimators = self.hyper_params['iso_num_estimators']
        iso_max_samples = self.hyper_params['iso_max_samples']
        iso_max_features = self.hyper_params['iso_max_feaures']

        trained_model = IsolationForest(n_estimators=iso_num_estimators, max_samples=iso_max_samples, max_features = iso_max_features,
                                   contamination=0.01)  # , random_state=rng) #estimators should be 200

        return trained_model

    def _create_enhanced_isolation_forest(self):

        iso_num_trees = self.hyper_params['iso_num_estimators']
        iso_max_samples = self.hyper_params['iso_max_samples']

        trained_model = EnhancedIsolationForest(num_trees = iso_num_trees, max_samples = iso_max_samples)

        return trained_model


    def finalise_graph(self, plt, plt_title):
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def plot_unsupervised_graphs(self):
        df_with_score = self.add_score_column(self.df_train_master, self.df_X_train_scaled)
        df_test_with_score = self.add_score_column(self.df_test_master, self.df_X_test_scaled)

        data = df_with_score['score']
        plt.title('Anomaly Scores - Train set')
        plt.ylabel('Frequency')
        plt.xlabel('Score')
        # plt.plot(data)
        plt.hist(data, bins=30)
        self.finalise_graph(plt, "Train set - Anomaly Scores")

        data = df_test_with_score['score']
        plt.title('Anomaly Scores - Test set')
        plt.ylabel('Frequency')
        plt.xlabel('Score')
        # plt.plot(data)
        plt.hist(data, bins=30)
        self.finalise_graph(plt, "Test set - Anomaly Scores")

    def plot_supervised_graphs(self):
        df_with_pred = self.add_prediction_column(self.df_train_master, self.df_X_train_scaled)
        df_test_with_pred = self.add_prediction_column(self.df_test_master, self.df_X_test_scaled)

        plt.plot(self.df_costs)  # x='Trn',y='Tst')
        self.finalise_graph(plt, "Cost per iteration")

        data = df_with_pred['perc_diff']
        plt.title('Train set % differences (actual vs predicted Tax) per employee')
        plt.ylim(top=30, bottom=-30)
        plt.ylabel('% difference')
        plt.xlabel('Employee')
        plt.plot(data)
        self.finalise_graph(plt, "Train set - Percentage diff per employee")



        data = df_with_pred['perc_diff']
        data_abs = df_with_pred['perc_diff'].apply(lambda x: abs(x))
        x_vals = [i for i in range(len(data_abs))]
        plt.title('Train set  abs % differences (actual vs predicted Tax) per employee')
        plt.ylim(top=30)
        plt.ylabel('abs % difference')
        plt.xlabel('Employee')
        plt.bar(x_vals, data_abs)
        self.finalise_graph(plt, "Train set  abs % differences (actual vs predicted Tax) per employee")


        data = df_with_pred['perc_diff']
        data_noinf = data[~data.isin([np.nan, np.inf, -np.inf])]
        data_clipped = data_noinf.apply(lambda x: min(50, x))
        data_clipped = data_clipped.apply(lambda x: max(-50, x))
        plt.title('Train set  % differences (actual vs predicted Tax) per employee Histogram')
        # plt.ylim(top=30)
        # plt.ylabel('% difference')
        # plt.xlabel('Employee')
        plt.hist(data_clipped, bins=50)
        self.finalise_graph(plt, "Train set - Percentage diff per employee Histogram")


        data = df_test_with_pred['perc_diff']
        plt.title('Test set % differences (actual vs predicted Tax) per employee')
        plt.ylim(top=30, bottom=-30)
        plt.ylabel('% difference')
        plt.xlabel('Employee')
        plt.plot(data)
        self.finalise_graph(plt, "Test set - Percentage diff per employee")


        green_diamond = dict(markerfacecolor='g', marker='D')
        data = df_test_with_pred['perc_diff']
        fig1, ax1 = plt.subplots()
        ax1.set_title('Test set: Percentage differences Box Plot')
        ax1.boxplot(data, flierprops=green_diamond)
        self.finalise_graph(plt, "Test set - Percentage differences")

        data = df_with_pred['perc_diff']
        fig1, ax1 = plt.subplots()
        ax1.set_title('Train set: Percentage differences Box Plot')
        ax1.boxplot(data, flierprops=green_diamond)
        self.finalise_graph(plt, "Train set - Percentage differences")


    def plot_graphs(self):

        if self.is_unsupervised():
            self.plot_unsupervised_graphs()
        else:
            self.plot_supervised_graphs()

    def train(self,trained_model_in=None,report_progress=True):
        """
        Main entry point.

        Determines algorithm to use based on ML Service and trains a model.

        If already-trained model supplied, uses this as a basis otherwise starts with new model.

        If supervised:
           - Adds prediction column to train and test master dfs once trained

        If unsupervised:
            - Adds score column to train and test master dfs once trained


        Returns:
            self.trained_model,avg_cost, avg_test_cost, acc_trn, acc_tst
        """

        if report_progress:
            print('\n\nTraining commencing')
        # if interface_to_azure:
        #     run = exp.start_logging()

        curr_epoch = 0
        num_iters = self.hyper_params['num_iters']

        if trained_model_in is None:
            self.trained_model = self._initialise_trained_model()
            # self.trained_model = MLPRegressor(random_state=4242, alpha=alpha, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes,
            #                   batch_size=batch_size, learning_rate_init=learning_rate_init) #, warm_start=True)  # (max_iter=200)
        else:
            self.trained_model = trained_model_in

        costs = []

        avg_cost, avg_test_cost, acc_trn, acc_tst, i = self._train_model(report_progress)

        if report_progress:
            print('Training complete')
        curr_epoch = i

        self.df_costs = pd.DataFrame(costs, columns=['Trn', 'Tst'])

        return self.trained_model,avg_cost, avg_test_cost, acc_trn, acc_tst


class AzureTrainer(Trainer):
    """
    Subclass of Trainer.

    Contains methods for logging results/stats to Azure

    """

    def __init__(self,ml_service,model_name,hyper_params=None,in_trained_model=None,in_scaler_X=None,in_scaler_Y=None,local_run_log_to_azure="N",local_log_experiment_name='exp', local_log_ws_config_path=None,base_folder=None,model_version=None,clip_training_set=None, clip_test_set=None):
        super(AzureTrainer, self).__init__(ml_service,model_name,hyper_params=hyper_params,in_trained_model=in_trained_model,in_scaler_X=in_scaler_X,in_scaler_Y=in_scaler_Y,base_folder=base_folder,model_version=model_version,clip_training_set=clip_training_set, clip_test_set=clip_test_set)

        self.local_run_log_to_azure = local_run_log_to_azure
        self.local_log_experiment_name = local_log_experiment_name

        self.run = Run.get_context()
        if hasattr(self.run, 'properties'):
            print('running on azure. Properties: ' + str(self.run.properties))
            self.run_type = 'run_on_azure'

        else:
            print('running locally')
            print('arg: ' + self.local_run_log_to_azure)
            if self.local_run_log_to_azure == 'Y':
                print('but logging to azure')
                self.run_type = 'run_local_log_azure'
                print('Using ws config path: ',local_log_ws_config_path)
                self.ws = Workspace.from_config(path=local_log_ws_config_path)
                print(self.ws.name, self.ws.location, self.ws.resource_group, self.ws.location, sep='\t')
                self.exp = Experiment(workspace=self.ws, name=self.local_log_experiment_name)
                self.run = self.exp.start_logging()

            else:
                print('no azure logging')
                self.run_type = 'local_only'

        # if run_type == 'run_local_only':
        if self.run_type == 'local_only':
            pass
        else:
            self.run.log('ML Service', self.ml_service)
            self.run.log('Model Name', self.model_name)
            # run.log('Model Vers', model_vers)
            start_time = datetime.datetime.now()
            self.run.log('Training start time', str(start_time))

    @staticmethod
    def trainer_from_args(args):
        print('hidden layers: ', args.hp_hidden_layers)

        data_folder = args.data_folder
        print('Data folder:', data_folder)

        model = args.model
        model_vers = args.model_version
        ml_service = args.ml_service
        clip_training_set = args.clip_training_set
        clip_test_set = args.clip_test_set

        if '.' in args.hp_iso_max_samples:
            iso_max_samples = float(args.hp_iso_max_samples)  # fraction of samples to use
        else:
            try:
                iso_max_samples = int(args.hp_iso_max_samples)  # number of samples to use
            except:
                iso_max_samples = args.hp_iso_max_samples  # string, eg 'auto'

        if '.' in args.hp_iso_max_features:
            iso_max_features = float(args.hp_iso_max_features)  # fraction of features to use
        else:
            iso_max_features = int(args.hp_iso_max_features)  # number of features  to use


        hyper_params = {'run_num': 1, 'learning_rate_init': args.hp_lr_init, 'alpha': args.hp_reg,
                        'hidden_layer_sizes': args.hp_hidden_layers, 'num_iters': args.hp_iters, 'max_iter': 50000,
                        'iso_num_estimators': args.hp_iso_num_estimators, 'iso_max_samples': iso_max_samples,
                        'iso_max_features': iso_max_features,
                        'batch_size': 200, 'clip_training_set':args.clip_training_set, 'clip_test_set': args.clip_test_set}

        local_run_log_to_azure = args.local_run_log_to_azure
        local_log_ws_config_path = args.local_log_ws_config_path
        local_log_experiment_name = args.local_log_experiment_name
        base_folder = args.data_folder

        return AzureTrainer(ml_service,model,hyper_params=hyper_params,local_run_log_to_azure=local_run_log_to_azure,local_log_experiment_name=local_log_experiment_name,local_log_ws_config_path=local_log_ws_config_path,base_folder=base_folder,model_version=model_vers,clip_training_set=clip_training_set)

    def train(self,trained_model_in=None,report_progress=True):
        """
        Main entry point.

        Calls super (Trainer) train method, then additionally logs stats to Azure experiment if logging requested

        """
        trained_model, avg_cost, avg_test_cost, acc_trn, acc_tst =  super(AzureTrainer, self).train(trained_model_in,report_progress)

        if self.run_type == 'local_only':
            pass
        else:
            self.run.log('regularization rate', np.float(self.hyper_params['alpha']))
            self.run.log('Acc train', np.float(acc_trn))
            self.run.log('Acc test', np.float(acc_tst))
            self.run.log('cost train', np.float(avg_cost))
            self.run.log('cost test', np.float(avg_test_cost))
            self.run.log('iters', int(self.hyper_params['num_iters']))
            self.run.log('batch size', int(self.hyper_params['batch_size']))
            self.run.log('learning rate', np.float(self.hyper_params['learning_rate_init']))
            self.run.log('num hidden layers', int(len(self.hyper_params['hidden_layer_sizes'])))
            self.run.log('iso num estimators', int(self.hyper_params['iso_num_estimators']))
            self.run.log('iso max samples', str(self.hyper_params['iso_max_samples']))
            self.run.log('iso max features',str(self.hyper_params['iso_max_features']))
            self.run.log('num train examples',int(self.df_X_train.shape[0]))
            trn_cost_only = self.df_costs['Trn']
            tst_cost_only = self.df_costs['Tst']
            self.run.log_list('Costs per iteration Trn', trn_cost_only, description='')
            self.run.log_list('Costs per iteration Tsts', tst_cost_only, description='')
            self.run.log('algorithm',self.get_learning_algorithm())
            self.run.log('training_code_version', Trainer.trainer_code_version)
            end_time = datetime.datetime.now()
            self.run.log('Training end time',str(end_time))

        return trained_model, avg_cost,avg_test_cost,acc_trn,acc_tst


    def write_model(self):

        """Write trained model to disk as .pkl

          File name is in format: <ml_service>_model_<model name>_<model version to train>.pkl

          Objects written to .pkl file consist of a single Python list containing:
              - trained model
              - scaler_X used
              - scaler_Y used
              - JSON model config

          Note: .pkl file is written to special "outputs" directory for Azure use
          """

        model_file_name = self.ml_service + '_model_' + self.model_name + '_' + self.model_version_to_train + '.pkl'
        model_full_file_name = os.path.join(self.finder.get_output_folder(), model_file_name)
        model_objects_list = [self.trained_model, self.scaler_X, self.scaler_Y, self.model_config.j_config]
        os.makedirs(self.finder.get_output_folder(), exist_ok=True)
        joblib.dump(value=model_objects_list, filename=model_full_file_name)

        #if self.run_type == 'run_on_azure':
        # Also save to outputs folder for auto save in Azure
        os.makedirs('outputs', exist_ok=True)
        # note file saved in the outputs folder is automatically uploaded into experiment record
        joblib.dump(value=model_objects_list, filename=os.path.join('outputs', model_file_name))
        print('\nwriting to outputs: ' + model_file_name)


    def finalise_graph(self, plt, plt_title):
            if self.run_type == 'local_only':
                pass
            else:
                self.run.log_image(plt_title, plot=plt)
            if self.run_type != 'run_on_azure':
                plt.show()
            plt.clf()
            plt.cla()
            plt.close()




if __name__ == '__main__':


    model =   'A001' # 'M010'# 'SS01' #'M010' # 'SS01' #'T001' #'T006'
    ml_service =  'PAD' # 'TWV' #'PAD'
    version = None  #'008' #None for curr vers

    base_folder = '../data'

    #model =  'MIN2'
    #ml_service = 'PAD'

    training_clip = -1
    trainer = AzureTrainer(ml_service,model,model_version = version, base_folder=base_folder, clip_training_set=training_clip, clip_test_set=10)
    trainer.hyper_params['num_iters'] =  5500 #150000
    trainer.hyper_params['hidden_layer_sizes'] =  [20,20] #400,400,400,400]
    trainer.hyper_params['learning_rate_init'] = 0.01 #0.01
    trainer.hyper_params['alpha'] =   1e-05 # 1000.0 # 1e-05 #100000.0 #1e-05 #1000.0 #1e-05

    trainer.hyper_params['iso_num_estimators'] = 100
    trainer.hyper_params['iso_max_samples'] = 'auto'

    #trainer.hyper_params['clip_training_set'] = 100
    # trainer.clip_training_set()

    trained_model, avg_cost, avg_test_cost, acc_trn, acc_tst = trainer.train(trained_model_in=None, report_progress=True)

    trainer.write_model()

    trainer.plot_graphs()

    trainer.run.complete()

    if trainer.is_unsupervised():
        df_with_important_features = trainer.determine_important_features(trainer.df_X_train_scaled, trainer.df_train_master, 3,
                                                                            show_progress_after_each=20,
                                                                            show_diagnostics_flag=True)

        df_sorted_with_score_col = trainer.add_score_column(df_with_important_features, trainer.df_X_train_scaled,
                                                              sort_by_scores=True)
        #df_sorted_with_score_col = df_with_important_features.copy()
        scores = df_sorted_with_score_col['score'].tolist()
        ret_json = {"Scores": scores}
        df_sorted_with_score_col.to_csv(os.path.join(base_folder,'temp/tr_with_pred_unsup_enhanced_with_new_analyser_class.csv'))
    else:
        df_with_pred_col = trainer.add_prediction_column(trainer.df_X_train, trainer.df_X_train_scaled)
        df_with_pred_col.to_csv(os.path.join(base_folder,'temp/tr_with_pred_sup.csv'))
    #df_train_with_pred = trainer.add_prediction_column(trainer.df_train_master,trainer.df_X_train_scaled)

    #df_test_with_pred = trainer.add_prediction_column(trainer.df_test_master, trainer.df_X_test_scaled)





    #df_test_with_pred.to_csv('te_with_pred.csv')



    if False:
        # old logic
        model_config = MLModelConfig.get_model_config_from_web_service_for_model(ml_service,model)
        period = 'MULTI'

        #df_train,df_test = read_train_test(j_config,target_vers = '001',period='MULTI')
        df_train_name = model_config.assemble_file_prefix(period,multi=True,data_dir = 'data/train') + '_formatted_train.csv'
        df_test_name =  model_config.assemble_file_prefix(period,multi=True,data_dir = 'data/train') + '_formatted_test.csv'


        df_train_master = pd.read_csv(df_train_name)
        df_test_master  = pd.read_csv(df_test_name)

        print('df train: ' + str(df_train_master.shape))
        print('df test: ' + str(df_test_master.shape))
        df_train_master.head()



        df_X_train,df_Y_train = prepare_dataset(df_train_master,model_config)
        df_X_test, df_Y_test = prepare_dataset(df_test_master, model_config)


    #    df_X_train_scaled, df_X_test_scaled, df_Y_train_scaled, df_Y_test_scaled,scaler_X,scaler_Y = normalise(df_X_train,df_X_test,df_Y_train,df_Y_test)
        df_X_train_scaled,df_Y_train_scaled, scaler_X,scaler_Y = normalise(df_X_train,df_Y_train)
        df_X_test_scaled,df_Y_test_scaled, scaler_X,scaler_Y = normalise(df_X_test,df_Y_test,scaler_in_X = scaler_X,scaler_in_Y=scaler_Y)



        param_runs = set_hyperparameters()

        run_order = [13]
        #run_order = [12]
        #run_order = [11]
        #run_order = [10]
        #run_order = [7,4,8,2,3,5,6,1]
        # run_order = [3,5,6,1]
        # run_order = [7,4,8,2,9,3,5,6,1]
        # run_order = [7,8]

        trained_model = None
        results = []
        for p in run_order:
            #nn_model = train(param_runs[p - 1],p,df_X_train_scaled,df_X_test_scaled,df_Y_train_scaled,df_Y_test_scaled,scaler_X,scaler_Y,True,False,None)
            trained_model,avg_cost, avg_test_cost, acc_trn, acc_tst,df_costs = train(param_runs[p-1],df_train_master,df_test_master,df_X_train_scaled, df_X_test_scaled, df_Y_train_scaled, df_Y_test_scaled,scaler_X, scaler_Y,model_config, trained_model_in=None,report_progress=True,interface_to_azure = False,exp=None,ws=None)
            results.append([p, avg_cost, avg_test_cost, acc_trn, acc_tst])

            model_p = trained_model.get_params()
            model_p['learning_rate_init'] = 0.0001
            trained_model.set_params(**model_p)
            trained_model, avg_cost, avg_test_cost, acc_trn, acc_tst, df_costs = train(param_runs[p - 1], df_train_master, df_test_master,
                                                                                  df_X_train_scaled, df_X_test_scaled,
                                                                                  df_Y_train_scaled, df_Y_test_scaled,
                                                                                  scaler_X, scaler_Y, model_config, trained_model_in=trained_model,
                                                                                  report_progress=True,
                                                                                  interface_to_azure=False, exp=None,
                                                                                  ws=None)
            results.append([p,avg_cost,avg_test_cost,acc_trn,acc_tst])
        #params = {'learning_rate_init': 0.1, 'alpha': 1.0, 'hidden_layer_sizes': [100], 'num_iters': 10000,
        # 'max_iter': 50000, 'batch_size': 200}
        #nn_model = train(params, 1, df_train,df_test,df_X_train_scaled, df_X_test_scaled, df_Y_train_scaled, df_Y_test_scaled,
        #                 scaler_X, scaler_Y, True, False, None)

        print (str(results))

        all_res = []
        for res in results:
            all_res.append([res[3],res[4]])
        print ('aha')

        df_costs.plot()
        plt.show()

        df_test_with_pred = add_prediction_column(trained_model, df_test_master, df_X_test_scaled, scaler_Y,model_config)

        green_diamond = dict(markerfacecolor='g', marker='D')
        data = df_test_with_pred['perc_diff']
        fig1, ax1 = plt.subplots()
        ax1.set_title('Percentage differences Box Plot')
        ax1.boxplot(data, flierprops=green_diamond)
        plt.show()

        df_train_with_pred = add_prediction_column(trained_model, df_train_master, df_X_train_scaled, scaler_Y,model_config)

        green_diamond = dict(markerfacecolor='g', marker='D')
        data = df_train_with_pred['perc_diff']
        fig1, ax1 = plt.subplots()
        ax1.set_title('Percentage differences train Box Plot')
        ax1.boxplot(data, flierprops=green_diamond)
        plt.show()