"""
    Pipeline related classes
        - PipelineProcess - base class for pipeline subclasses below
        - EUHDumpFormatter - [**Deprecated**] Pipeline class to manage pre-processing of raw euHReka files to format into required input format (temporary only - euHReka PCC should send correct file format ready for routing)
        - FileRouter - Pipeline class to manage routing of files sent by payroll services to correct folder for relevant ml service/model
        - PipelineTransformer - Pipeline class to transform collection of input files  for an ml service/model in preparation for training of a model (create training/test sets, normalise data etc). Also used in predictions, ie to transform prediction data into correct format

    Instances of PipelineProcess subclasses above can be run stand-alone for testing, or run online within  Azure ML Service (where source code has been uploaded).

    Pipeline  flow is as follows:

    1) route data (route payroll training data received from payroll services for various ml services/models  into correct Azure folder. Uses pipeline_step_route_data.py (instantiates FileRouter)
    2) transform data (transform training data, ie standardise, OHE etc,  for an ml service/model/version  and build a training set. Uses pipeline_step_transform_data.py (instantiates PipelineTransformer)
    3) train model (see train module)

    Relevant python source code modules (used both for stand-alone use and for upload to Azure) are:
        - pipeline_step_format_euHReka_dump.py [**Deprecated**]
        - pipeline_step_route_data.py
        - pipeline_step_transform_data.py


    Helper classes:
        - FileFinder - base class used to manage input/output folder/file names
        - EUHDumpFinder - [**Deprecated**] subclass specifically for euHReka raw files

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ngamlfpy.utils import df_to_csv,find_in_list_partial, pulled_json_to_df, conv_value_title_list_to_dict
from ngamlfpy.hrxmlconfig import MLModelConfig
import os
import csv
import shutil
from sklearn.externals import joblib

class FileFinder():
    """
    Class to manage input and output folders

    Public methods:

    - get_input_folder
    - get_output_folder
    - get_processed_folder
    - get_input_file_names: list all input files in input folder
    - parse_input_file_name(file_name): split input file name into identifying source attributes
    """

    MLFOLDER_BASE_FOLDER = './data'

    MLFOLDER_INPUT = 'input'
    MLFOLDER_INPUT_SUFFIX = '_input.csv'

    MLFOLDER_TRANSFORMED = 'train'

    MLFOLDER_PREDICT = 'predict'

    MLFOLDER_PROCESSED = 'processed'


    def __init__(self, ml_service, use_model_name_in=False, use_model_name_out=False, model_name=None, model_version = None, base_folder=None, relative_input_folder = MLFOLDER_INPUT, relative_output_folder=MLFOLDER_INPUT, relative_processed_folder=MLFOLDER_PROCESSED, output_file_suffix=''):

        """
        :param ml_service:  ML Service, eg PAD or TWV
        :param use_model_name_in: If supplied, appends model name to input path (and version if version supplied)
        :param use_model_name_out: If supplied, appends model name to output path (and version if version supplied)
        :param model_name: Model name, eg T001
        :param model_version: Model version
        :param base_folder: Base folder for all data, eg ./data
        :param relative_input_folder: Input folder name, eg input/ or train/ etc
        :param relative_output_folder:  Output folder name, eg /train etc
        :param relative_processed_folder:  Folder to store copy of processed input files
        :param output_file_suffix:  Suffix to append to output files
        """

        self.ml_service = ml_service
        self.model_name = model_name
        self.model_version = model_version
        self.relative_input_folder = relative_input_folder
        self.relative_output_folder = relative_output_folder
        self.relative_processed_folder = relative_processed_folder
        self.output_file_suffix = output_file_suffix
        self.use_model_name_in = use_model_name_in
        self.use_model_name_out = use_model_name_out

        if base_folder is None:
            self.base_folder = FileFinder.MLFOLDER_BASE_FOLDER
        else:
            self.base_folder = base_folder


    def __ml_service_path(self):
        ml_service_path = os.path.join(self.base_folder,self.ml_service)
        return ml_service_path

    def __add_ml_service_to_path(self,path):
        ml_service_path = self.__ml_service_path()
        return os.path.join(ml_service_path,path)


    def __add_model_to_path(self,path):
        ml_service_path = self.__ml_service_path()
        ml_service_plus_path = os.path.join(ml_service_path,path)
        if self.model_name is None:
            return ml_service_plus_path
        else:
            if self.model_version is None:
                return os.path.join(ml_service_plus_path,self.model_name)
            else:
                return os.path.join(os.path.join(ml_service_plus_path, self.model_name),self.model_version)

    def get_input_folder(self):
        if self.use_model_name_in:
           return self.__add_model_to_path(self.relative_input_folder)
        else:
           return  self.__add_ml_service_to_path(self.relative_input_folder)

    def get_processed_folder(self):
        return self.__add_ml_service_to_path(self.relative_processed_folder)

    def get_output_folder(self):
        if self.use_model_name_out:
            return self.__add_model_to_path(self.relative_output_folder)
        else:
            return self.__add_ml_service_to_path(self.relative_output_folder)

    def get_output_file_suffix(self):
        return self.output_file_suffix

    def get_input_file_names(self):

        print('Searching for files in: ', self.get_input_folder())
        try:
            file_list = os.listdir(self.get_input_folder())
        except:
            print(self.get_input_folder() + ' folder does not exist')
            return []

        files_only = []
        for file_name in file_list:
            if os.path.isdir(os.path.join(self.get_input_folder(), file_name)):
                pass
            else:
                files_only.append(file_name)


        return files_only


    def get_full_input_file_name(self, file_name):
        return os.path.join(self.get_input_folder(), file_name)


    def parse_input_file_name(self,file_name,include_remainder=False):
        """
        Expected file name structure:
            - Source attributes delimited by '_'
            - First 8 attributes expected are:
                - ML Service
                - Model
                - Model Version
                - Source Payroll Service
                - Source GCC
                - Source LCC
                - Model Variant
                - Source System
            - Other free format attributes can be appended after first 8 (also delimited by '_')
        """
        if include_remainder:
            return file_name.split('_')[:8] + ['_'.join(file_name.split('_')[8:])]
        else:
           return file_name.split('_')[:8]

    def assemble_output_file_name_prefix_from_selection(self,selection):
        '''
        Currently not used (used only in experimental FilePoster class)
        '''
        return "_".join([selection['system'],selection['client'],selection['gcc'],selection['lcc'],selection['payroll_area'],selection['period']])


class EUHDumpFinder(FileFinder):
    """
    **Deprecated** - euHReka file format now as per other ML services. Use  FileFinder instead.

    Subclass of FileFinder to specifically manage raw euHReka input and output folders

    Public methods:

    - get_input_folder
    - get_output_folder
    - get_processed_folder
    - get_input_file_names (list all input files in input folder)
    """

    MLFOLDER_RAW_EUHREKA_INPUT_PENDING = 'raw_input/pending'
    MLFOLDER_RAW_EUHREKA_INPUT_PROCESSED = 'raw_input/processed'
    MLFOLDER_RAW_EUHREKA_INPUT_IT_PREFIX = 'IT'
    MLFOLDER_RAW_EUHREKA_INPUT_WT_PREFIX = 'WTS'

    def __init__(self, ml_service, base_folder=None):
        super(EUHDumpFinder, self).__init__(ml_service, base_folder=base_folder, relative_input_folder = EUHDumpFinder.MLFOLDER_RAW_EUHREKA_INPUT_PENDING, relative_output_folder=FileFinder.MLFOLDER_INPUT,
                     relative_processed_folder=EUHDumpFinder.MLFOLDER_RAW_EUHREKA_INPUT_PROCESSED,
                     output_file_suffix=FileFinder.MLFOLDER_INPUT_SUFFIX)

    def get_customer_list(self, up_to_col=6):
        file_list = self.get_input_file_names()

        file_parts = [file_name.split('_')[:up_to_col] for file_name in file_list]

        return file_parts

    def check_if_list_equal(self, l1, l2):

        for i, entry in enumerate(l1):
            if entry != l2[i]:
                return False

        return True

    def get_unique_file_sets(self):

        file_parts_no_detail = self.get_customer_list()

        found_parts = []

        for file_part in file_parts_no_detail:

            found = False
            for found_part in found_parts:
                if self.check_if_list_equal(found_part, file_part):
                    found = True
                    break
            if not found:
                found_parts.append(file_part)

        return found_parts

    def get_file_prefix_for_set(self, unique_file_set):
        system, client, gcc, lcc, payroll_area, period = unique_file_set
        return '_'.join([system, client, gcc, lcc, payroll_area, period])

    def get_wt_full_file_name(self,unique_file_set):

        prefix = self.get_file_prefix_for_set(unique_file_set)
        file_name = prefix + '_' +  EUHDumpFinder.MLFOLDER_RAW_EUHREKA_INPUT_WT_PREFIX + '.txt'
        return os.path.join(self.get_input_folder(), file_name)


    def get_infotype_file_names(self,unique_file_set):
        all_files = self.get_input_file_names()

        it_files = []
        prefix = self.get_file_prefix_for_set(unique_file_set)
        prefix += '_' + EUHDumpFinder.MLFOLDER_RAW_EUHREKA_INPUT_IT_PREFIX
        for file_name in all_files:
            if file_name[:len(prefix)] == prefix:
                it_files.append(file_name)
        return it_files



class PipelineProcess:
    '''
    Base class for ML for PY pipeline processes. Essentially an abstract class - only subclasses are instantiated.

    Provides the following services:
      - Manages input and output files based on FileFinder object
      - Logs key attributes to Azure experiment

    '''

    pipeline_code_version = '0.1g' # Used within Azure ML Service to determine which code version used

    def __init__(self, ml_service,base_folder=None,predict=False):
        self.ml_service = ml_service

        self.predict=predict

        self.finder = FileFinder(ml_service,base_folder)

        self.files_processed = []

        self.move_raw_input_files_once_processed = True

        self.get_azure_run_context()


    def move_raw_input_files_to_processed(self):

        if self.predict:
            pass
        else:
            if os.path.exists(self.finder.get_processed_folder()):
                pass
            else:
                os.mkdir(self.finder.get_processed_folder(), 755)
            for file_name in self.files_processed:
                path, filename_part = os.path.split(file_name)
                shutil.move(file_name, self.finder.get_processed_folder())

    # def get_output_file_prefix(self):
    #
    #     return self.input_file_prefix

    def write_output_file(self, df, output_file_prefix='out_'):

        if self.predict:
            pass
        else:
            output_folder = self.finder.get_output_folder()
            if os.path.exists(output_folder):
                pass
            else:
                os.mkdir(output_folder,755)

           # output_file_prefix = self.get_output_file_prefix() #self.input_file_prefix  # '_'.join([self.model_config.get_model_name(),self.input_file_prefix])
            file_name = os.path.join(output_folder,
                                     output_file_prefix)  # '/'.join([self.formatted_folder,output_file_prefix])
            file_name += self.finder.get_output_file_suffix()  # MLFOLDER_RAW_EUHREKA_INPUT_PREFORMATTED_SUFFIX
            df_to_csv(df, file_name)


    def remove_dups(self,seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def get_azure_run_context(self):

        from azureml.pipeline.core import PipelineRun
        self.run = PipelineRun.get_context()
        if hasattr(self.run, 'properties'):
            self.run_type = 'run_on_azure'
            print('Running on Azure')
            print('p run: ', self.run)
            self.exp = self.run._experiment
            print('exp: ', self.run._experiment)

            self.ws = self.exp._workspace

            # ws = Workspace.get(args.workspace_name,subscription_id='0c7f1ae6-7228-403f-ba54-84e72c46d6cf')
            print('ws: ', self.ws)
            self.ds = self.ws.get_default_datastore()
            print('ds: ', self.ds)
        else:
            self.run_type = 'run_local_only'
            print('Running locally')

        self.run.log('pipeline_runtype', self.run_type)
        self.run.log('pipeline_code_version', PipelineProcess.pipeline_code_version)

        return self.run

class EUHDumpFormatter(PipelineProcess):
    '''

    **Deprecated** - euHReka now sending files in same format as other payroll services, so  FileRouter is now the first pipeline step used for all ml services

    Pipeline process class to manage formatting raw euHReka files into single row format files required by ML Framework.

    Usage: instantiate EUHDumpFormatter object and call process_data()

    For use within Azure ML Service: refer pipeline_step_format_euHReka_dump.py (source code uploaded into Azure Pipeline)

    Makes use of EUHDumpFinder class to manage input file set names
    '''

    def __init__(self,ml_service,base_folder=None,suppress_scramble=False,only_remove_retro_rows=False):
        """
        :param ml_service: ML service, eg PAD, TWV etc
        :param base_folder:  Base folder for files, eg ./data
        :param suppress_scramble: Suppress scrambling of employee numbers. Default False
        :param only_remove_retro_rows: For employees with retros, only remove the actual retro rows, not the current period row. Default False
        """

        super(EUHDumpFormatter,self).__init__(ml_service,base_folder=base_folder)
        self.finder = EUHDumpFinder(ml_service,base_folder=base_folder)
        self.model_config = None # Only initialised once Infotype 1 data is read
        self.suppress_scramble = suppress_scramble
        self.only_remove_retro_rows=only_remove_retro_rows


    def get_unique_sets(self):
        ''' Note: Can handle multiple sets of input files found in input folder'''

        return self.finder.get_unique_file_sets()

    def process_file_set(self, unique_file_set):

        return None


    def replace_european_format(self,str):

        new_str = str.replace('.','')
        new_str = new_str.replace(',','.')
        return new_str



    def format_float_col(self,col):

         col_new  = col.apply(lambda x: str(x).replace(',', '') if ((len(str(x)) < 3) or  (str(x)[-3] == '.')) else self.replace_european_format(str(x)))
         col_new = col_new.astype(float)
         return col_new


    def remove_emps_with_retros(self,df):

        emps_with_retros = []

        if ('Period' in df.columns) and ('ForPeriod' in df.columns):
            for key, row in df.iterrows():
                if str(row['Period'])  == str(row['ForPeriod']):
                    pass
                else:
                    emps_with_retros.append(row['PERNR'])

            emps_with_retros = list(set(emps_with_retros))

            df_new = df[~df['PERNR'].isin(emps_with_retros)]
            num_removed = df.shape[0] - df_new.shape[0]
            print('\nNum emps with retros removed: ' + str(num_removed))
            return df_new
        else:
            print('InPeriod / ForPeriod col(s) not found. No emps with retros removed')
            return df


    def remove_retro_rows(self,df):

        if ('Period' in df.columns) and ('ForPeriod' in df.columns):
            df_new = df[df['Period'] == df['ForPeriod']]

            num_removed = df.shape[0] - df_new.shape[0]
            print('Num retro records removed: ',num_removed)
            return df_new
        else:
            print('InPeriod / ForPeriod col(s) not found. No retro records removed')
            return df

    def drop_unimportant_fields(self,df):

        unimportant_fields = ['','HISTO','ITXEX','REFEX','ORDEX','ITBLD','PREAS','FLAG1','FLAG2','FLAG3','FLAG4','RESE1','RESE2','OBJPS','SPRPS','SEQNR','AEDTM','UNAME','GRPVL']

        unimportant_fields += ['Last name First name','Name of Employee or Applicant','Name per. parameter','Name per. parameter.1','Payroll area text','Payroll area text.1','Personnel Area Text']

        strip_cols = []
        for col in df.columns:
            if col in unimportant_fields:
               strip_cols.append(col)

        df_new = df.drop(strip_cols,axis=1)
        return df_new


    def drop_duplicates(self,df):

        return df.drop_duplicates(subset='PERNR', keep="last")

    def drop_unused_fields(self,df):

        all_fields = self.model_config.get_all_fields()
        all_source_fields = [x['field'] for x in all_fields]

        strip_cols = []

        for col in df.columns:
            if col in all_source_fields:
                pass
            else:
                strip_cols.append(col)

        df_new = df.drop(strip_cols, axis=1)
        return df_new


    def apply_exclusion_groups(self,df):

        df_new = df.copy()

        exclusions = self.model_config.get_exclusions()
        for exclusion_set in exclusions:
            exclusion_1 = exclusion_set[0]
            exclusion_2 = exclusion_set[1]
            #TODO - cater for > 2 exclusions in a set (eg PERSG == 2 AND PERSK = 03 AND WERKS = 2001)
            df_new = df_new[~((df_new[exclusion_1['field']] == exclusion_1['value'])
                               & (df_new[exclusion_2['field']] == exclusion_2['value']))]

        num_removed = df.shape[0] - df_new.shape[0]
        print('Num records excluded via exclusion groups: ',num_removed)

        return df_new


    def read_se16_dump(self,file_name):
        full_name = os.path.join(self.finder.get_input_folder(), file_name)
        # print ('full_name: ' + full_name)

        row_list = []
        header = []

        print('Reading file: ' + full_name)

        with open(full_name,encoding='latin1') as csvfile:
            reader = csv.reader(csvfile, delimiter='\n')
            header_found = False
            for i,row in enumerate(reader):
                if len(row) == 0:
                    continue

                row_tabbed = row[0].split('\t')
                if len(row_tabbed) == 1: #try commas
                    row_tabbed = row[0].split(',')
                if len(row_tabbed) == 0:
                    pass
                elif row_tabbed[1] == 'MANDT' or row_tabbed[1] == 'PERNR':
                    header = row_tabbed[:]
                    header_found = True
                elif row_tabbed[0] == 'MANDT' or row_tabbed[0] == 'PERNR':
                    header = row_tabbed[:]
                    header_found = True
                elif (not header_found): # junk before header
                    pass

                else:
                    row_list.append(row_tabbed[:])
        self.files_processed.append(full_name)
        return header, row_list

    def read_se16_dumps(self):

        #TODO ensure I0001 is first file read (so filter on ABKRS can occur

        #mf_field_list = self.model_config.master_fields_per_infotype()

        #for i, itype_dets_key in enumerate(mf_field_list):
        for i,file_name in enumerate(self.infotype_file_names):
            #fields = mf_field_list[itype_dets_key]
            header, row_list = self.read_se16_dump(file_name)
            if i == 0:
                df = pd.DataFrame(row_list, columns=header[:len(row_list[0])])
                df = self.drop_unimportant_fields(df)
                df = df[df.YYGCC == self.gcc] #ensure file only contains what it says it does
                df = df[df.YYLCC == self.lcc] #ensure file only contains what it says it does
                if self.payroll_area is None:
                    pass
                else:
                    df = df[df.ABKRS == self.payroll_area] #ensure file only contains what it says it does


                df = df.drop_duplicates(subset=['PERNR'],keep='last')

                self.model_config = MLModelConfig.get_model_config_from_web_service_for_cust(self.ml_service,
                                                                                             self.system,
                                                                                             self.gcc,
                                                                                             self.lcc,
                                                                                             payroll_area=self.payroll_area)

                if self.model_config is None:
                    raise(Exception('Error retrieving model config from web service: ' + self.ml_service + ' ' + self.system + ' ' + self.gcc + ' ' + self.lcc + ' ' + self.payroll_area))

                df = self.drop_unused_fields(df)
                #df = df[fields]


            else:
                df_new = pd.DataFrame(row_list, columns=header[:len(row_list[0])])
                df_new = self.drop_unimportant_fields(df_new)
                df_new = self.drop_duplicates(df_new)
                if 'MANDT' in df_new.columns:
                    df_new = df_new.drop(columns=['MANDT']) # already in Infotype 1
                df_new = self.drop_unused_fields(df_new)
                #df_new = df_new[['PERNR'] + fields]


                df = pd.merge(df, df_new, on='PERNR', how='left')

        return df

    def rename_cols(self, df):

        col_trans = {}

        for feature in self.model_config.get_all_fields():
            if ('conv' not in feature) or (feature['conv'] == ''):
                if feature['title'] in df.columns:
                    pass  # already column with new name
                else:
                    col_trans[feature['field']] = feature['title']

        df_new = df.rename(columns=col_trans)

        return df_new

    # def rename_cols(self,df):
    #
    #     col_trans = {}
    #
    #     for feature in self.model_config.j_config['modelfeatures']:
    #         col_trans[feature['field']] = feature['title']
    #
    #     df_new = df.rename(columns=col_trans)
    #
    #     return df_new


    def derive_group(self):

        group = MLModelConfig.get_group_for_payroll_area(self.gcc,self.lcc,self.payroll_area,self.ml_service)
        return group



    def add_derived_features(self, df):

        new_df = df.copy()
        derived_feats_added = []

        for feat in self.model_config.get_all_fields():
            # if feat == 'Year':
            if 'conv' in feat:
                if feat['conv'] == '':
                    pass
                else:
                    if feat['conv'][:2] == 'CH':
                        # character subset
                        st = int(feat['conv'].split('[')[1].split('-')[0]) - 1
                        end = int(feat['conv'].split('-')[1].split(']')[0])
                        source_field_title = self.model_config.get_title_for_field(feat['field'])
                        new_col = df[source_field_title].apply(lambda x: str(x)[st:end])
                        new_df[feat['title']] = new_col.values
                        derived_feats_added.append(feat['title'])
                    elif feat['conv'][:2] == 'CS':
                        # constant
                        val = feat['conv'].split('[')[1].split(']')[0]
                        #new_col = df[source_field_title].apply(lambda x: str(x)[st:end])
                        new_col = pd.Series([val for i in range(df.shape[0])])
                        new_df[feat['title']] = new_col.values
                        derived_feats_added.append(feat['title'])
                    elif feat['conv'][:2] == 'VA':
                        # Get Group from Payroll Area
                        source_field_title = self.model_config.get_title_for_field(feat['field'])
                        group = self.derive_group()
                        new_col = pd.Series([group for i in range(df.shape[0])])
                        new_df[feat['title']] = new_col.values
                        derived_feats_added.append(feat['title'])
                    elif feat['conv'][:2] == 'MI':
                        # Modelinfo field from model config
                        field_name = feat['conv'].split('[')[1].split(']')[0]
                        field_val = self.model_config.j_config[field_name]
                        new_col = pd.Series([field_val for i in range(df.shape[0])])
                        new_df[feat['title']] = new_col.values
                        derived_feats_added.append(feat['title'])


        print('\nDerived features added: ' + str(derived_feats_added))
        return new_df

    def order_df(self, df):

        curr_cols = list(df.columns)

        all_model_field_names = self.model_config.get_all_field_names()

        new_cols = []

        for model_field_name in all_model_field_names:
            fields_in_curr_cols = find_in_list_partial(model_field_name, curr_cols)
            new_cols.extend(fields_in_curr_cols)

        new_cols = self.remove_dups(new_cols)

        df_reordered = df[new_cols]

        return df_reordered

    def add_missing_wt_cols(self, df):
        # Add columns of all zeros for any features not existing in wt reporter dump

        df_new = df.copy()

        col_trans = {}

        rt_wt_features = self.model_config.rt_wt_features()

        missing_wt_cols = []

        for feature in rt_wt_features:
                if feature['title'] in df.columns:
                    pass  # already column  exists
                else:
                    missing_wt_cols.append(feature['title'])


        for missing_wt_col in missing_wt_cols:
            new_col = pd.Series([0.0 for i in range(df.shape[0])])
            df_new[missing_wt_col] = new_col.values

        print('Number of missing WT feature columns (now added with all zeroes): ' + str(len(missing_wt_cols)))
        return df_new

    def columnise_wt_dump(self,df):

        grouped = df.groupby(['PERNR', 'FPPER'])

        num_processed = 0
        #wts_requested = self.model_config.rt_wts()

        #df_new = pd.DataFrame(columns=list(df.columns) + wts_requested)

        columnised_list = []
        for key, entries in grouped:

            if (num_processed % 500 == 0):
                print('Records processed: ',num_processed,' / ',len(grouped))
            wts = entries.groupby(['WT']).sum()['Amount']
            #entry = entries.iloc[0].copy()
            entry = entries.iloc[0].to_dict()

            for wt, amt in wts.iteritems():
                wt_col = 'WT_' + wt
                entry[wt_col] = amt

            #df_new = df_new.append(entry)
            columnised_list.append(entry)

            num_processed += 1

        df_new =   pd.DataFrame(columnised_list)
        df_new = df_new.fillna(0)

        df_new = df_new.drop(columns=['WT', 'Amount'])
        if 'ABKRS' in df_new.columns:
            df_new = df_new.drop(columns=['ABKRS'])


        return df_new



    def read_wt_reporter_dump(self,unique_file_set,delimiter='\t'):
        col_translations = {'PY Area, IP': 'ABKRS', 'Pers.No.': 'PERNR',
                            '                            Amount': 'Amount', 'PA': 'PersArea'}

        #full_name = '_'.join([self.full_prefix, MLFOLDER_RAW_EUHREKA_INPUT_WT_PREFIX]) + '.txt'
        full_name = self.finder.get_wt_full_file_name(unique_file_set)
        # wt_list_wts = []
        # for feature in self.model_config.j_config['modelfeatures']:
        #     spl = feature['field'].split('WT_')
        #     if len(spl) > 1:
        #         wt_list_wts.append(spl[1])
        #
        # for feature in self.model_config.j_config['modellabels']:
        #     spl = feature['field'].split('WT_')
        #     if len(spl) > 1:
        #         wt_list_wts.append(spl[1])

        print('Reading: ' + full_name)

        df = pd.read_csv(full_name, delimiter=delimiter,
                         dtype={'Pers.No.': str, 'PERNR':str,'In-period': str,'In-Period':str,'For-period': str, 'WT': str},
                                #'Amount':str,'                            Amount': str},
                                encoding='latin1')

        df = self.drop_unimportant_fields(df)

        df = df.rename(columns=col_translations)

        #df['Amount'] = self.format_float_col(df['Amount'])

        #df = df[df.WT.isin(wt_list_wts)]

        #df = df[df['In-period'] == period]  # j['selection']['period']]
        #df = df[df['ABKRS'] == self.payroll_area]

        #all_rt_fields = self.model_config.rt_fields(excl_wts=True)
        #cols = ['PERNR'] + all_rt_fields + ['WT', 'Amount']

        #df = df[cols]

        self.files_processed.append(full_name)
        return df

    def get_output_file_prefix(self):
        payroll_service = 'EUH'
        return '_'.join([self.ml_service,self.model_config.get_model_name(),self.model_config.get_model_version(),payroll_service,self.gcc,self.lcc,self.derive_group(),self.system,self.client,self.payroll_area,self.period])


    def scramble_pernr(self,pernr):
        pernr_str = str(pernr).zfill(8)
        pernr_str_adj = ''
        for i in range(0,len(pernr_str)):
            new_digit = int(pernr_str[i]) + i + 1
            pernr_str_adj += str(new_digit)[-1]
        pernr_str_adj_2 = ''
        for i in range(0,4):
            new_digit = int(pernr_str_adj[i]) + int(pernr_str_adj[i+4])
            pernr_str_adj_2 += str(new_digit)[-1]
        for i in range(4,len(pernr_str)):
            new_digit = int(pernr_str_adj[i]) + int(pernr_str_adj_2[i-4])
            pernr_str_adj_2 += str(new_digit)[-1]
        transl = {'1':'k','3':'C','5':'e','7':'J','0':'t'}

        pernr_str_adj_3 = ''
        for i in range(0,len(pernr_str)):
            if pernr_str_adj_2[i] in transl:
               pernr_str_adj_3 += transl[pernr_str_adj_2[i]]
            else:
               pernr_str_adj_3 += pernr_str_adj_2[i]



        return pernr_str_adj_3



    def scramble_pernrs(self,df):
        df_new = df.copy()
        if 'PERNR' in df_new.columns:
            df_new['PERNR']  = df_new['PERNR'].apply(self.scramble_pernr)

        return df_new

    def process_file_set(self,unique_file_set):
        '''
        Process unique file set.

        - Read all masterfile Infotype dumps (eg I1 plus any other optional infotypes)
        - Restrict to exclude employees in exclusion groups
        - Read WT Reporter pay results extract and columnise (ie 1 record per emp with column per wage type)
        - Combine master and pay data into 1 row per emp
        - Rename columns into names as per ML Config
        - Add zero columns for wt columns required but not found in input
        - Remove retros as required
        - Add derived columns (as per ML Config)
        - Sort columns into order specified by ML Config


        '''

        print(' ')
        print('Processing file set: ',unique_file_set)

        # self.model_config = MLModelConfig.get_model_config_from_web_service_for_cust(self.ml_service,self.system,self.gcc,self.lcc,payroll_area=self.payroll_area)

        self.infotype_file_names = self.finder.get_infotype_file_names(unique_file_set)

        self.md_df = self.read_se16_dumps()
        self.md_df = self.md_df.replace(np.nan, '', regex=True)

        self.md_df = self.apply_exclusion_groups(self.md_df)

        if self.gcc == 'SOL' and self.lcc == 'SFR':
            delim = ','
        else:
            delim = '\t'  # use default
        self.py_df = self.read_wt_reporter_dump(unique_file_set, delimiter=delim)
        self.py_df_col = self.columnise_wt_dump(self.py_df)
        self.py_df_col  = self.drop_unused_fields(self.py_df_col)

        self.combined_df = pd.merge(left=self.md_df, right=self.py_df_col, on='PERNR')
        # self.combined_df = self.rename_cols(self.combined_df)


        self.combined_df = self.rename_cols(self.combined_df)

        self.combined_df = self.add_missing_wt_cols(self.combined_df)

        if self.only_remove_retro_rows:
            self.combined_df = self.remove_retro_rows(self.combined_df)  # only remove retro records (retain current records even if retros exist for emp)
        else:
            self.combined_df = self.remove_emps_with_retros(self.combined_df)  # completely remove emps who have retros

        self.combined_df = self.add_derived_features(self.combined_df)

        self.combined_df = self.order_df(self.combined_df)

        return self.combined_df



    def process_data(self):
        ''' Process all input file sets'''

        unique_file_sets = self.finder.get_unique_file_sets()

        output_df = None

        if unique_file_sets is None:
            print('No processing')
            print('Num files processed: ',len(self.files_processed))

        else:
            for unique_file_set in unique_file_sets:
                self.files_processed = []
                output_file_prefix = self.finder.get_file_prefix_for_set(unique_file_set)
                self.system, self.client, self.gcc, self.lcc, self.payroll_area, self.period = unique_file_set

                output_df = self.process_file_set(unique_file_set)

                if self.suppress_scramble:
                    pass
                else:
                    output_df = self.scramble_pernrs(output_df)

                if output_df is None:
                    print('No processing for: ', unique_file_set)
                else:
                    self.write_output_file(output_df, output_file_prefix=output_file_prefix)
                    if self.move_raw_input_files_once_processed:
                          self.move_raw_input_files_to_processed()

            print(' ')
            print('Num file sets processed: ',len(unique_file_sets))

        self.output_df = output_df
        return output_df



class PipelineTransformer(PipelineProcess):
    '''
      Pipeline process class to manage:
       - transformation of training input files to build a training/test set for input into ML training
       - transformation of prediction input files prior to prediction

      Usage: instantiate PipelineTransformer object and call process_data()

      For use within Azure ML Service: refer pipeline_step_transform_data.py (source code uploaded into Azure Pipeline)

      Makes use of FileFinder class to manage input and output file names



      Summary:

      - Removes unwanted columns (ie columns not appearing in mlconfig) and renames columns as specified in mlconfig
      - Removes retros
      - One Hot Encodes categorical features
      - Re-order columns into order specified in mlconfig
      - Split data into training and test sets (if predict flag not set)
      - Prepare dataset
      - Normalise (scale) data
      - If predict flag not set, write output files to /train/<model>/<vers>/ folder (as training set), ready for training




      '''



    def __init__(self,ml_service,base_folder=None,model_name=None,model_version=None,in_json = None,in_df=None,predict=False,normalise_labels=False, scaler_X = None, scaler_Y = None):
        """

        :param ml_service: ML Service, eg TWV or PAD
        :param base_folder: eg ./data
        :param model_name: eg M003
        :param model_version: Optional. If not specified, use latest model version available
        :param in_json: Optional (used normally as input for predictions)
        :param in_df:  Optional (used normally  as input for training sets)
        :param predict:  Flag indicating whether input is intended for predictions
        :param normalise_labels: Flag indicating whether to normalise (scale) labels as well as features (default False)
        :param scaler_X: If supplied, scale features using this supplied scaler object (otherwise calculate scaling)
        :param scaler_Y: If supplied, scale labels using this supplied scaler object (otherwise calculate scaling, if normalise labels flag is True)
        """

        super(PipelineTransformer, self).__init__(ml_service,base_folder=base_folder,predict=predict)
        self.move_raw_input_files_once_processed = False
        self.in_json = in_json
        self.in_df = in_df
        self.model_name = model_name
        self.normalise_labels = normalise_labels
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.predict = predict


        if in_json is not None:
            sel_json = self.in_json['selection']
            sel_json = conv_value_title_list_to_dict(sel_json)
            sel_json['system'] = sel_json['System']
            sel_json['gcc'] = sel_json['GCC']
            sel_json['lcc'] = sel_json['LCC']
            if 'Variant' in sel_json:
                sel_json['variant'] = sel_json['Variant']
            if 'ABKRS' in sel_json:
                sel_json['payroll_area'] = sel_json['ABKRS']
            sel_json['ml_service'] = sel_json['MLService']

            #selection = [sel_json['system'], '', sel_json['gcc'], sel_json['lcc'], sel_json['payroll_area'], '']
        elif in_df is not None:
            sel_json = self.get_selection_details_from_first_df_record()
            #selection = [sel_json['system'], '', sel_json['gcc'], sel_json['lcc'], sel_json['payroll_area'], '']


        if (in_json is None) and (in_df is None):

            cust_data = MLModelConfig.get_sample_customer_using_model(self.ml_service,self.model_name)
            self.model_config = MLModelConfig.get_model_config_from_web_service_for_cust(self.ml_service,cust_data['system'],cust_data['gcc'],cust_data['lcc'],cust_data['variant'])
        else:
            self.model_config = MLModelConfig.get_model_config_from_web_service_for_cust(self.ml_service,sel_json['system'],sel_json['gcc'],sel_json['lcc'],payroll_area=None if not 'payroll_area' in sel_json else sel_json['payroll_area'], group=None if not 'variant' in sel_json else sel_json['variant'])
            self.model_name = self.model_config.get_model_name()
            #(in_json['selection']['ml_service'],in_json['selection']['system'],in_json['selection']['gcc'],in_json['selection']['lcc'],payroll_area=  in_json['selection']['payroll_area'])                                                                                                     '])

        if model_version is None:
           self.model_version_to_transform =  self.model_config.get_model_version() # use latest
        else:
            self.model_version_to_transform = model_version

        self.finder = FileFinder(ml_service,use_model_name_in=True,use_model_name_out=True,model_name=self.model_name,model_version=self.model_version_to_transform,base_folder=base_folder,relative_input_folder=FileFinder.MLFOLDER_INPUT,relative_output_folder=FileFinder.MLFOLDER_TRANSFORMED,output_file_suffix=None)



    def remove_retros(self,df):
            if ('Period' in df.columns) and ('ForPeriod' in df.columns):
                  df_new = df[df['Period'] == df['ForPeriod']]
                  num_removed = df.shape[0] - df_new.shape[0]
                  print('\nNum retros removed: ' + str(num_removed))
                  return df_new
            else:
                print('InPeriod / ForPeriod col(s) not found. No retros removed')
                return df


    def rename_cols(self, df):

        col_trans = {}

        for feature in self.model_config.get_all_fields():
            if ('conv' not in feature) or (feature['conv'] == ''):
               if feature['title'] in df.columns:
                   pass # already column with new name
               else:
                   col_trans[feature['field']] = feature['title']

        df_new = df.rename(columns=col_trans)


        return df_new


    def filter_unwanted_columns(self, df):

        model_fields = self.model_config.get_all_field_names()
        matching_fields = []
        for field_name in model_fields:
            if field_name in df.columns:
                matching_fields.append(field_name)
            else:
                print('Field: ', field_name, ' not found')

        df_new = df[matching_fields]

        return df_new

    def read_input_file(self,file_name):

        full_name = self.finder.get_full_input_file_name(file_name)
        df = pd.read_csv(full_name)

        return df, full_name



    def convert_string_fields(self,df):

        all_fields = self.model_config.get_all_fields()

        new_df = df.copy()
        for field in all_fields:
            if 'data_type' in field:
                if field['data_type'] == 'S':
                    if field['title'] in new_df.columns:
                        if new_df[field['title']].dtype == object:
                            pass
                        else:
                            new_df[field['title']] = new_df[field['title']].fillna(0)
                            new_df[field['title']] = new_df[field['title']].astype(str)

        return new_df

    def format_data(self,df):

        print('Source data in: ',df.shape)
        df_renamed = self.rename_cols(df)

        df_filtered = self.filter_unwanted_columns(df_renamed)
        df_filtered = self.convert_string_fields(df_filtered)

        print ('Filtered: ',df_filtered.shape)
        return df_filtered


    def collect_and_format_data(self):
        #cust_data = MLModelConfig.get_sample_customer_using_model(self.ml_service, self.model_name)
        #self.model_config = MLModelConfig.get_model_config_from_web_service_for_cust(self.ml_service,
                                                                                     #cust_data['system'],
                                                                                     #cust_data['gcc'], cust_data['lcc'],
                                                                                     #cust_data['variant'])
        print('\nSearching for source files for model: ',self.model_name, ' version: ',self.model_version_to_transform)
        all_files = self.finder.get_input_file_names()
        print(len(all_files), ' files found')

        df_cons = None

        for file_name in all_files:
            ml_service, model_name, model_vers, payroll_service, gcc, lcc, group, rest = self.finder.parse_input_file_name(
                file_name)
            if model_vers == self.model_version_to_transform:

                df,full_name = self.read_input_file(file_name)
                df_formatted = self.format_data(df)
                if df_cons is None:
                    df_cons = df_formatted
                else:
                    df_cons = df_cons.append(df_formatted)


        return df_cons


    def get_selection_details_from_first_df_record(self):
        first_rec = self.in_df.iloc[0]
        if 'PayrollArea' in self.in_df.columns:
            pa = first_rec['PayrollArea']
        else:
            pa = first_rec['ABKRS']
        sel = {'system': first_rec['System'], 'gcc': first_rec['GCC'], 'lcc': first_rec['LCC'],'variant':first_rec['Variant'],'payroll_area': pa}

        # optional fields
        if 'Period' in self.in_df.columns:
            sel['Period'] = first_rec['Period']

        if 'Client' in self.in_df.columns:
            sel['Client'] = first_rec['Client']

        return sel

    def order_df(self,df):

        curr_cols = list(df.columns)


        all_model_field_names = self.model_config.get_all_field_names()

        new_cols = []

        for model_field_name in all_model_field_names:
            fields_in_curr_cols = find_in_list_partial(model_field_name, curr_cols)
            new_cols.extend(fields_in_curr_cols)

        new_cols = self.remove_dups(new_cols)

        df_reordered = df[new_cols]

        return df_reordered

    def one_hot_encode(self,df, cat_feature):
        new_df = df.copy()

        num_cats = 0

        cat_df = pd.get_dummies(df[cat_feature], prefix=cat_feature)
        new_df = pd.concat([new_df, cat_df], axis=1)
        new_df = new_df.drop([cat_feature], axis=1)
        num_cats = cat_df.shape[1]

        return new_df, num_cats

    def one_hot_encode_all_cat_features(self,df):

        df_ohe = df.copy()
        cat_feats = self.model_config.get_categorical_features()

        all_num_cats = []

        for cat_feat in cat_feats:
            df_ohe, num_cats = self.one_hot_encode(df_ohe, cat_feat)
            all_num_cats.append(num_cats)

        print('\nCategorical features one hot encoded: ' + str(cat_feats) + ' Number categories: ' + str(all_num_cats))
        return df_ohe, all_num_cats


    # def add_derived_features(self,df):
    #
    #     new_df = df.copy()
    #     derived_feats_added = []
    #
    #     for feat in self.model_config.get_all_fields():
    #         # if feat == 'Year':
    #         if 'conv' in feat:
    #             if feat['conv'] == '':
    #                 pass
    #             else:
    #                 if feat['conv'][:2] == 'CH':
    #                     # character subset
    #                     st = int(feat['conv'].split('[')[1].split('-')[0]) - 1
    #                     end = int(feat['conv'].split('-')[1].split(']')[0])
    #                     source_field_title = self.model_config.get_title_for_field(feat['field'])
    #                     new_col = df[source_field_title].apply(lambda x: str(x)[st:end])
    #                     new_df[feat['title']] = new_col
    #                     derived_feats_added.append(feat['title'])
    #
    #     print('\nDerived features added: ' + str(derived_feats_added))
    #     return new_df

    def split(self,df, test_size=0.2, random_state=42, use_ascending_gross_order=False):
        if use_ascending_gross_order:
            df_sorted = df.sort_values(by='TaxableGross')
            num_test_examples = float(df.shape[0]) * test_size
            num_train_examples = int(df.shape[0] - num_test_examples)
            df_train = df_sorted[:num_train_examples]
            df_test = df_sorted[num_train_examples:]

        else:
            df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        return df_train, df_test

    def prepare_dataset(self,df, gross_only=False):

#        num_key_cols = self.model_config.get_num_sel_and_key_cols(list(df.columns))
        selection_field_names = self.model_config.get_selection_field_names()
        selection_field_names_found_in_df = []

        data_source_field_names = self.model_config.get_data_source_field_names()
        data_source_field_names_found_in_df = []

        cols = df.columns
        for sel_field_name in selection_field_names:
            if sel_field_name in cols:
                selection_field_names_found_in_df.append(sel_field_name)
        for data_source_field_name in data_source_field_names:
            if data_source_field_name in cols:
                data_source_field_names_found_in_df.append(data_source_field_name)



        label_col = self.model_config.get_label_col()

        if gross_only:
            df_X = df[['TaxableGross']]
        else:
            df_X =  df.drop(selection_field_names_found_in_df,axis=1)#df.drop(df.iloc[:, :num_key_cols], axis=1)
            df_X = df_X.drop(data_source_field_names_found_in_df, axis=1)  # df.drop(df.iloc[:, :num_key_cols], axis=1)
            if label_col is None: #unsupervised
                pass
            else:
                if self.predict:
                   if label_col in df_X.columns:
                       df_X = df_X.drop([label_col], axis=1)
                else:
                    df_X = df_X.drop([label_col], axis=1)

        if self.predict:
            df_Y = None
        else:
            if label_col is None: #unsupervised
                df_Y = None
            else:
                df_Y = df[[label_col]]

        return df_X, df_Y

    def normalise(self,df_X, df_Y, scaler_in_X=None, scaler_in_Y=None, scale_Y=False):

        if scaler_in_X is not None:
            scaler_X = scaler_in_X
            df_X_scaled = scaler_X.transform(df_X)
        else:
            scaler_X = MinMaxScaler()
            df_X_scaled = scaler_X.fit_transform(df_X)
        df_X_scaled = pd.DataFrame(df_X_scaled, columns=df_X.columns)

        if scale_Y:
            if df_Y is None:
                df_Y_scaled = None
                scaler_Y = scaler_in_Y
            else:
                if scaler_in_Y is not None:
                    scaler_Y = scaler_in_Y
                    df_Y_scaled = scaler_Y.transform(df_Y)
                else:
                    scaler_Y = MinMaxScaler()
                    df_Y_scaled = scaler_Y.fit_transform(df_Y)
                df_Y_scaled = pd.DataFrame(df_Y_scaled, columns=df_Y.columns)
        else:

            scaler_Y = scaler_in_Y
            df_Y_scaled = df_Y

        return df_X_scaled, df_Y_scaled, scaler_X, scaler_Y

    def process_data(self):

        if self.in_df is not None:
            df = self.in_df
            df_formatted = self.format_data(df)
        elif self.in_json is not None:
            df = pulled_json_to_df(self.in_json,use_value_title_format=True)
            df_formatted = self.format_data(df)
        else:
            df_formatted = self.collect_and_format_data()

        print('Formatted: ',df_formatted.shape)

        df_transformed = self.remove_retros(df_formatted)
        print('Cleaned:', df_transformed.shape)

        # df_transformed = self.add_derived_features(df_transformed)
        # print('Added derived:', df_transformed.shape)

        df_transformed,all_num_cats = self.one_hot_encode_all_cat_features(df_transformed)
        print('OHE:', df_transformed.shape)

        df_transformed = self.order_df(df_transformed)
        print('\nOrdered:', df_transformed.shape)

        if self.predict:
            df_X_predict, _ = self.prepare_dataset(df_transformed)
            df_X_predict_scaled, _, scaler_X, _ = self.normalise(df_X_predict,None,scaler_in_X = self.scaler_X,scale_Y=self.normalise_labels)
        else:
            files_to_write = []

            if MLModelConfig.is_unsupervised(self.ml_service):
                test_size = 0.001
            else:
                test_size = 0.2

            df_train_master, df_test_master = self.split(df_transformed, test_size=test_size,use_ascending_gross_order=False)
            print('\nTrain: ' + str(df_train_master.shape) + ' Test: ' + str(df_test_master.shape))
            files_to_write.append({'suffix':'_train_master.csv','type':'dataframe','object': df_train_master})
            files_to_write.append({'suffix': '_test_master.csv', 'type':'dataframe','object': df_test_master})

            df_X_train, df_Y_train = self.prepare_dataset(df_train_master)
            df_X_test, df_Y_test = self.prepare_dataset(df_test_master)
            print('\nX Train: ' + str(df_X_train.shape))
            if df_Y_train is None:
                print(' No Y Train')
            else:
                print (' Y Train: ' + str(df_Y_train.shape))
            print('X Test: ' + str(df_X_test.shape))
            if df_Y_test is None:
                print(' No Y Test')
            else:
                print(' Y Test: ' + str(df_Y_test.shape))
            files_to_write.append({'suffix': '_X_train.csv', 'type':'dataframe','object': df_X_train})
            files_to_write.append({'suffix': '_X_test.csv', 'type': 'dataframe', 'object': df_X_test})
            if df_Y_test is None:
                pass
            else:
                files_to_write.append({'suffix': '_Y_train.csv', 'type':'dataframe','object': df_Y_train})
                files_to_write.append({'suffix': '_Y_test.csv', 'type': 'dataframe', 'object': df_Y_test})



            df_X_train_scaled,df_Y_train_scaled,scaler_X, scaler_Y = self.normalise(df_X_train, df_Y_train,scale_Y = self.normalise_labels)
            df_X_test_scaled, df_Y_test_scaled, scaler_X, scaler_Y = self.normalise(df_X_test, df_Y_test, scaler_in_X=scaler_X,
                                                                               scaler_in_Y=scaler_Y,scale_Y = self.normalise_labels)
            print('X Train scaled: ' + str(df_X_train_scaled.shape))
            if df_Y_train_scaled is None:
                print(' No Y Train scaled')
            else:
                print(' Y Train scaled: ' + str(df_Y_train_scaled.shape))

            print('X Test scaled: ' + str(df_X_test_scaled.shape))
            if df_Y_test_scaled is None:
                print(' No Y Test scaled')
            else:
                print(' Y Test scaled: ' + str(df_Y_test_scaled.shape))

            files_to_write.append({'suffix': '_X_train_scaled.csv', 'type':'dataframe','object': df_X_train_scaled})
            files_to_write.append({'suffix': '_X_test_scaled.csv', 'type': 'dataframe', 'object': df_X_test_scaled})
            if df_Y_train_scaled is None:
                pass
            else:
                files_to_write.append(
                    {'suffix': '_Y_train_scaled.csv', 'type': 'dataframe', 'object': df_Y_train_scaled})
                files_to_write.append({'suffix': '_Y_test_scaled.csv', 'type': 'dataframe', 'object': df_Y_test_scaled})
            files_to_write.append({'suffix': '_scaler_X_model.pkl', 'type': 'scaler', 'object': scaler_X})

            if df_Y_train is None:
                pass
            else:
                files_to_write.append({'suffix': '_scaler_Y_model.pkl', 'type': 'scaler', 'object': scaler_Y})

            self.write_output_files(files_to_write)

        self.transformed_df = df_transformed

        if self.predict:
            return df_transformed, df_X_predict_scaled
        else:
            return df_transformed, None


    def write_output_files(self,files_to_write):

        os.makedirs(self.finder.get_output_folder(), exist_ok=True)

        output_file_prefix = self.model_name

        for file_data in files_to_write:
            file_name = os.path.join(self.finder.get_output_folder(),
                                     output_file_prefix)  # '/'.join([self.formatted_folder,output_file_prefix])
            file_name += file_data['suffix']  # MLFOLDER_RAW_EUHREKA_INPUT_PREFORMATTED_SUFFIX
            print('\nWriting: ' + file_name)

            if file_data['type'] == 'dataframe':
                 df_to_csv(file_data['object'], file_name)
            elif file_data['type'] == 'scaler':
                joblib.dump(file_data['object'], file_name)



class FileRouter(PipelineProcess):
    '''
         Pipeline process class to manage routing of  input files sent by payroll services to the correct folder (ml service/model) ready for transformation to build a training set using the PipelineTransformer class

         Usage: instantiate FileRouter object and call route_files()

         For use within Azure ML Service: refer pipeline_step_route_data.py (source code uploaded into Azure Pipeline)

         Makes use of FileFinder class to manage input and output directory/file names
         '''

    def __init__(self,ml_service,base_folder=None):
        """
        :param ml_service: Machine Learning Service, eg PAD or TWV
        :param base_folder: Base folder for data, eg ./data
        """

        super(FileRouter, self).__init__(ml_service,base_folder=base_folder)

        self.finder = FileFinder(ml_service, use_model_name_in=False, use_model_name_out=True, model_name=None,
                                 base_folder=base_folder, relative_input_folder=FileFinder.MLFOLDER_INPUT,
                                 relative_output_folder=FileFinder.MLFOLDER_INPUT,
                                 output_file_suffix=FileFinder.MLFOLDER_INPUT_SUFFIX)

        self.run.log('pipeline_searchfolder', self.finder.get_input_folder())

    def route_files(self):
        all_pending_files = self.finder.get_input_file_names()
        print('files to route: ',all_pending_files)
        for file_name in all_pending_files:

#            system,client,gcc,lcc,payroll_area,period,_ = self.finder.parse_input_file_name(file_name)
            ml_service,model_name,model_vers,payroll_service,gcc,lcc,group,rest = self.finder.parse_input_file_name(file_name)
            system = rest.split('_')[0]

            model_conf = MLModelConfig.get_model_config_from_web_service_for_cust(self.ml_service,system,gcc,lcc,group=group)#payroll_area = payroll_area)
            self.finder.model_name = model_conf.get_model_name()
            self.finder.model_version = model_vers
            output_folder = self.finder.get_output_folder()
            if os.path.exists(output_folder):
                pass
            else:
                os.makedirs(output_folder,755)
            path, filename_part = os.path.split(file_name)
            input_folder = self.finder.get_input_folder()
            full_path = os.path.join(input_folder,file_name)
            out_full_path = os.path.join(self.finder.get_output_folder(),file_name)
            if os.path.exists(out_full_path):
                print('File already exists: ',out_full_path,' - bypassing')
            else:
                shutil.move(full_path, self.finder.get_output_folder())

        self.run.log('numfilesrouted', len(all_pending_files))
        return all_pending_files

class FilePoster(PipelineProcess):
    ''' Experimental only - currently not in use'''

    def __init__(self,ml_service,in_json,base_folder=None):
        super(FilePoster, self).__init__(ml_service,base_folder=base_folder)
        self.in_json = in_json

        self.finder = FileFinder(ml_service, use_model_name_in=False, use_model_name_out=False, model_name=None,
                                 base_folder=base_folder, relative_input_folder=None,
                                 relative_output_folder=FileFinder.MLFOLDER_INPUT,
                                 output_file_suffix=FileFinder.MLFOLDER_INPUT_SUFFIX)

        self.run.log('pipeline_outputfolder', self.finder.get_output_folder())

    def write_output_file(self, df,full_output_name):

        df_to_csv(df, full_output_name)

    def post_json(self):

        file_name = self.finder.assemble_output_file_name_prefix_from_selection(self.in_json['selection'])
        file_name += self.finder.get_output_file_suffix()
        full_output_name = os.path.join(self.finder.get_output_folder(),file_name)

        df = pulled_json_to_df(self.in_json)

        self.write_output_file(df,full_output_name)

        return full_output_name




if __name__ == '__main__':

    print ('Running')

    if True:
        # test FileFinder
        ff = FileFinder('TWV',use_model_name_in=True, use_model_name_out=True, model_name='SS01',model_version='008', relative_input_folder='input', relative_output_folder='outp', relative_processed_folder='proc')
        print(ff.get_input_folder())
        print(ff.get_output_folder())
        print(ff.get_processed_folder())
        print(ff.get_input_file_names())
        f = ff.get_input_file_names()[0]
        print(ff.parse_input_file_name(f))

        feuh = EUHDumpFinder('PAD')
        print(feuh.get_input_folder())
        print(feuh.get_output_folder())
        print(feuh.get_processed_folder())
        print(feuh.get_input_file_names())



    if False:
        df = pd.read_csv('data/TWV/predict/T001/EQ1_402_ZCS_Z10_X1_201901_preformatted.csv')
        model_conf = MLModelConfig.get_model_config_from_web_service_for_cust('TWV','EQ1','ZCS','Z10',payroll_area='X1')

        json_dat = pulled_df_to_json(df,model_conf,'201901')
        with open('tst.json', 'w') as json_file:
            json.dump(json_dat, json_file,indent=4)

    PP_EUHDump_Format = 1
    PP_Route = 2
    PP_Transform = 3
    PP_Predict = 4




    ml_service = 'TWV' #'PAD' #'TWV'


    if False:
        file_router = FileRouter(ml_service)
        all_files = file_router.route_files()


    pipeline_training_processes =  [2] #[1,2,3] #[1,2]

    pipeline_predict_processes =[] #[3,4]


    if PP_EUHDump_Format in pipeline_training_processes:
        edFormatter = EUHDumpFormatter(ml_service)
        _ = edFormatter.process_data()

        print('Dump formatter complete')


    if PP_Route in pipeline_training_processes:
        file_router = FileRouter(ml_service)
        all_files = file_router.route_files()
        # formatter = PipelineFormatter(ml_service)
        # _ = formatter.process_data()

        print('Routing complete. ',len(all_files),' routed')

    model = 'T003'  # 'T006' #'MIN2' #'T001'
    if PP_Transform in pipeline_training_processes:
        # with open('data/TWV/predict/T001/tst_preformatted.json') as json_file:
        #     in_json = json.load(json_file)
        transformer = PipelineTransformer(ml_service,model_name=model) # in_json=in_json)
        transformed_df,_ = transformer.process_data()
        print('Transformer complete')



    if PP_Transform in pipeline_predict_processes:
        #file_name = 'TWV_EUH_ZCS_Z10_Mth_EQ1_402_X1_201902_preformatted.csv'
        file_name = 'PAD_EUH_INO_IFR_All_EP5_000_F0_201808_preformatted.csv'
        full_name = 'data/' + ml_service + '/predict/' + model + '/' + file_name
        scaler_X_name = model + '_scaler_X_model.pkl'
        scaler_X_full_name = 'data/' + ml_service + '/train/' + model + '/' + scaler_X_name
        in_df = pd.read_csv(full_name)
        scaler_X = joblib.load(scaler_X_full_name)
        transformer = PipelineTransformer(ml_service,model_name=model,in_df=in_df,predict=True,scaler_X=scaler_X) # in_json=in_json)
        transformed_predict_df,df_X_predict_scaled = transformer.process_data()
        print('Predict Transformer complete')

    if PP_Predict in pipeline_predict_processes:
            from ngamlfpy.train import AzureTrainer
            trainer = AzureTrainer(ml_service, model,load_already_trained_model=True)
            if trainer.is_unsupervised():
                df_with_score = trainer.add_score_column(transformed_predict_df,df_X_predict_scaled)
            else:
                df_with_pred = trainer.add_prediction_column(transformed_predict_df,df_X_predict_scaled,pred_set=True)

            print('Predict complete')

    # if PP_Format in pipeline_predict_processes:
    #     with open('data/TWV/predict/T001/tst_preformatted.json') as json_file:
    #         in_json = json.load(json_file)
    #     formatter = PipelineFormatter(ml_service,in_json=in_json,predict=True)
    #     formatted_df = formatter.process_data()
    #
    #     print('Formatter complete')



    if False:
       pass

       #  periods = ['201901', '201902']
       #  for i, period in enumerate(periods):
       #      j_data,resp_status = get_pay_data_from_web_service('EQ1', '402', 'ZCS', 'Z10', 'X1', period)
       #      if i == 0:
       #          df = pulled_json_to_df(j_data)
       #      else:
       #          df_new = pulled_json_to_df(j_data)
       #          df = df.append(df_new)
       #  df.head()
       #
       # # j_data = get_pay_data_from_web_service('EQ1', '402', 'ZCS', 'Z10', 'X1', period)
       #
       #
       #  # Remove retros
       #  df_no_retros = remove_retros(df)
       #
       #  # with open('data/sample_pull_config.json') as json_data:
       #  #    j_config = json.load(json_data)
       #  model_config = ModelConfig.get_model_config_from_web_service_for_cust('EQ1','402','ZCS','Z10','X1')
       #
       #  # Add derived features
       #  df_with_derived = add_derived_features(df_no_retros,model_config)
       #
       #  # One Hot Encode
       #  df_ohe,all_num_cats = one_hot_encode_all_cat_features(df_with_derived,model_config)
       #
       #  # Order dataset
       #  df_ordered = order_df(df_ohe,model_config)
       #
       #  period = 'MULTI'
       #
       #  # save interim
       #  df_to_csv(df_ordered,model_config.assemble_file_prefix(period,data_dir = 'data/train') + '_formatted_all.csv')
       #
       #  # Split to train and test set
       #  df_train,df_test = split(df_ordered)
       #
       #  # save train, test
       #  df_to_csv(df_train,model_config.assemble_file_prefix(period,data_dir = 'data/train') + '_formatted_train.csv')
       #  df_to_csv(df_test, model_config.assemble_file_prefix(period, data_dir='data/train') + '_formatted_test.csv')

    print ('Pipeline complete')