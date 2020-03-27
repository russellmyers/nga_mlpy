'''

Script for deployed prediction web service code.

Main entry points are init() and run(raw_data), as per Azure specification

raw_data is  the json prediction input  posted to the prediction  web service

'''


import json
from sklearn.externals import joblib
from ngamlfpy.utils import pulled_json_to_df,  conv_value_title_list_to_dict
from ngamlfpy.hrxmlconfig import MLModelConfig
from ngamlfpy.pipeline import PipelineTransformer
from ngamlfpy.train import AzureTrainer
from azureml.core.model import Model
from azureml.core import Run


run_type =  'run_on_azure'  # 'run_on_azure' #'run_local_only'
code_version = '0.1h' #Increment on each change to code


def get_run_type():

    # TODO Fix  below to cater for local or azure runs (seems run.get_context() isnt working in predict web service)
    global run_type
#     run = Run.get_context()
#     if hasattr(run, 'properties'):
#             print('running on azure. Properties: ' + str(run.properties))
#             run_type = 'run_on_azure'
#     else:
#             print('running locally only')
#             run_type = 'run_local_only'
#
#
#     print('Overriding - running on Azure')
#     run_type = 'run_on_azure'

    return run_type


def get_model(model_code,ml_service,model_version):
    global model, nn_model,scaler_X,scaler_Y,j_config_from_model

    run_type = get_run_type()

    if run_type == 'run_local_only':
        # downloaded model name:
        model_name =  ml_service + '_model_'+ model_code + '_' + model_version
        model_file_name = 'data/' + ml_service + '/train/'  + model_code + '/' + model_version + '/' +  model_name + '.pkl' #'twv_dnn_model_'+ model_code  + '.pkl'
        model_objects_list = joblib.load(model_file_name)
    else:
        # Published model deployed to prediction web service
        model_name =  ml_service + '_model_' + model_code + '_' + model_version   #'twv_dnn_' + model_code
        model_path = Model.get_model_path(model_name)  #('twv_dnn_' + model_code) # +  '_' + target_vers + '_run' + str(run_num))
        model_objects_list = joblib.load(model_path)
        
    nn_model = model_objects_list[0]
    scaler_X = model_objects_list[1]
    scaler_Y = model_objects_list[2]
    j_config_from_model = model_objects_list[3]  
    
    return model_name
    

def init():
    pass
    # Can't retrieve model yet since depends on data supplied in json within run method


def run(raw_data):
    global model_nn,scaler_X,scaler_Y,j_config_from_model

    use_value_title_format = True

    j_data = json.loads(raw_data)['data']
    print('j_data: ' +str(j_data)[:100])
    df_in = pulled_json_to_df(j_data,use_value_title_format=use_value_title_format)
    print ('df in shape: ' + str(df_in.shape))
    if 'TaxExIndicator' in df_in.columns:
        df_in['TaxExIndicator'] = df_in['TaxExIndicator'].fillna(0)
    if 'client' in  df_in.columns:
        df_in['client'] = df_in['client'].astype(str)
    if 'period' in df_in.columns:
        df_in['period'] = df_in['period'].astype(str)
    if 'ForPeriod' in df_in.columns:
        df_in['ForPeriod'] = df_in['ForPeriod'].astype(str)
    if 'PERNR' in df_in.columns:
        df_in['PERNR'] = df_in['PERNR'].astype(str)
    
    sel_dict = j_data['selection']
    if use_value_title_format:
        sel_dict = conv_value_title_list_to_dict(sel_dict)

    web_service_call_successful = False
    model_config = MLModelConfig.get_model_config_from_web_service_for_cust(ml_service=sel_dict['MLService'],
                                                                                              gcc=sel_dict['GCC'],
                                                                                              lcc=sel_dict['LCC'],
                                                                                              system=sel_dict['System'],
                                                                                              group = sel_dict['Variant'] if 'Variant' in sel_dict else None,
                                                                                              payroll_area = sel_dict['ABKRS'] if 'ABKRS' in sel_dict else None)
    if  model_config is not None:    #config_ws_call_status == 200:
        model_code = model_config.get_model_name() #j_config['model_metadata']['model']
        ml_service = model_config.selection['ml_service']
        model_version = model_config.get_model_version()
        web_service_call_successful = True
    else:
        model_code = 'T001' #default if error
        ml_service = 'TWV'
        model_version = '001'
        j_config = j_config_from_model
        model_config = MLModelConfig(j_config)

    model_name = get_model(model_code,ml_service,model_version)

    transformer = PipelineTransformer(ml_service=ml_service,base_folder=None,model_name=model_code,in_json = j_data,in_df=None,predict=True,normalise_labels=False, scaler_X = scaler_X, scaler_Y = scaler_Y)
    transformed_df,df_X_pred_scaled = transformer.process_data()

    predicter = AzureTrainer(ml_service=ml_service,model_name=model_code,hyper_params=None,in_trained_model=nn_model,in_scaler_X=scaler_X,in_scaler_Y=scaler_Y,local_run_log_to_azure="N",local_log_experiment_name='exp', local_log_ws_config_path=None,base_folder=None)

    if predicter.is_unsupervised():
        df_with_important_features = predicter.determine_important_features(df_X_pred_scaled, transformed_df,
                                                                            num_top_features=3,
                                                                            show_progress_after_each=20,
                                                                            show_diagnostics_flag=True)

        df_with_score_col = predicter.add_score_column(df_with_important_features, df_X_pred_scaled,
                                                              sort_by_scores=True)
        scores_and_important_features = df_with_score_col[['PERNR','score','rank','F1','F2','F3','LF1','LF2','LF3']].values.tolist()
        out_list = [{'PERNR':x[0],'Score':x[1],'Rank':x[2],'Important_Features':x[3:6]} for x in scores_and_important_features]

    else:
        df_with_pred_col = predicter.add_prediction_column(transformed_df,df_X_pred_scaled)

        y_actual = None
        label_col = model_config.get_label_col()
        if label_col in list(df_with_pred_col.columns):
            y_actual = df_with_pred_col[label_col].tolist()

        y_hat = df_with_pred_col['pred'].tolist()
        # make prediction
        # you can return any data type as long as it is JSON-serializable

        pred_columns = ['PERNR','pred']
        if label_col in df_with_pred_col.columns:
            pred_columns.append(label_col)

        predictions = df_with_pred_col[pred_columns].values.tolist()
        if label_col in df_with_pred_col.columns:
            out_list = [{'PERNR':x[0],'Pred':x[1],'Actual':x[2]} for x in predictions]
        else:
            out_list = [{'PERNR': x[0], 'Pred': x[1]} for x in predictions]

    ret_json = {}
    ret_json['selection'] = j_data['selection']
    ret_json['info'] = {}
    ret_json['info']['config_web_service_call_status'] = 'Success' if web_service_call_successful else 'Fail'
    ret_json['info']['azure_model_name'] = model_name
    ret_json['info']['model_code'] = model_code
    ret_json['info']['model_version'] = model_version
    ret_json['info']['pipeline_code_version'] = PipelineTransformer.pipeline_code_version
    ret_json['info']['predict_code_version'] = code_version
    ret_json['Predictions'] = out_list

    return ret_json    


run_type = get_run_type()
