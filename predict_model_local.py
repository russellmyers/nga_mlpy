'''

Script to perform predictions locally (not via web service, but using same code as web service  uses, ie calls predict_model.py code)
Uses local trained model instead of models registered in Azure


'''


import json
from predict_model import init,run
import predict_model
import argparse
import os
import pandas as pd
from ngamlfpy.pipeline import FileFinder
from ngamlfpy.hrxmlconfig import MLModelConfig
from ngamlfpy.utils import pulled_df_to_json

parser = argparse.ArgumentParser()
parser.add_argument('--ml_service', type=str, dest='ml_service', default='TWV', help='ML Service, eg TWV')
parser.add_argument('--model', type=str, dest='model', default='T001', help='Model code eg T001')
parser.add_argument('--data_folder', type=str, dest='data_folder', default='./data', help='Data folder, eg ./data')
parser.add_argument('--model_version', type = str, dest = 'model_version', default = '001', help = 'Model version eg 001')
parser.add_argument('--clip_emps',type= int, dest = 'clip_emps', default=None,help='Number of emps to include in prediction. Uses all emps if this param is not supplied')
parser.add_argument('--input_file_name', type = str,dest='input_file_name', help = 'Input file name, eg TWV_SS01_011_EUH_SOL_SUS_BIWK_PP9_100_U3_201915_input.csv' )
parser.add_argument('--output_predict_json_as_file', dest='output_predict_json_as_file', action='store_true', help = 'Flag indicating whether to also  output the json used for prediction as a file')



args = parser.parse_args()

model = args.model

finder = FileFinder(args.ml_service, use_model_name_in=True, use_model_name_out=False, model_name=args.model, model_version = args.model_version,
                         base_folder=args.data_folder, relative_input_folder=FileFinder.MLFOLDER_INPUT,
                         relative_output_folder=FileFinder.MLFOLDER_PREDICT)

full_path_in =  finder.get_full_input_file_name(args.input_file_name)  #os.path.join(args.data_folder, args.ml_service + '/input/' + args.model + '/' + args.model_version)

#full_path_in = os.path.join(folder,args.input_file_name)

if full_path_in.split('.')[-1] == 'json':
    with open(full_path_in) as json_file:
        j_predict = json.load(json_file)
else:

    df = pd.read_csv(full_path_in)
    #df = pd.read_csv('data/PAD/input/' + model_code + '/PAD_EUH_AKN_A05_ALL_EP5_984_N0_201908_input.csv')
    #df = pd.read_csv('data/PAD/input/M005/001/PAD_M005_001_EUH_ZZZ_Z10_MTHLY_EDF_310_ZZ_201905_input.csv')
    print(df.head())

    file_name_parsed = finder.parse_input_file_name(args.input_file_name,include_remainder=True)
    #_,_,_,payroll_service, gcc, lcc, group, system, rest = finder.parse_input_file_name(args.input_file_name,include_remainder=True)
    client, abkrs, period, other = file_name_parsed['rest'].split('_')
    ml_config = MLModelConfig.get_model_config_from_web_service_for_cust(args.ml_service, system=file_name_parsed['system'],gcc=file_name_parsed['gcc'],lcc=file_name_parsed['lcc'],payroll_area=abkrs)

    j_predict = pulled_df_to_json(df, ml_config, period, use_first_data_line_as_selection=True, use_value_title_format=True,
                                  values_only=False, clip_emps=args.clip_emps)

#outJson = input_json_file_name
#print('Writing out json file: ', outJson)



#predict_file_name =  model + '_test.json'
#predict_file_name = 'PAD_EUH_AKN_A05_ALL_EP5_984_N0_201908.json'

predict_model.run_type = 'run_local_only'

init()


# if predict_source == 'json':
#     with open(full_predict_file_name) as json_data:
#         j_data = json.load(json_data)

raw_data = {}
raw_data['data'] = j_predict

if args.output_predict_json_as_file:
    predict_json_input_file_name =  args.input_file_name.split('.')[0] + '.json'
    full_path_json_in = os.path.join(finder.get_output_folder(),predict_json_input_file_name)
    print('Writing json predict in file: ', full_path_json_in)
    with open(full_path_json_in, 'w') as outfile:
        json.dump(raw_data, outfile, indent=4)

res = run(json.dumps(raw_data))
print ('res web service all status: ' + str(res['info']['config_web_service_call_status']))
print ('res Azure model name used: ' + str(res['info']['azure_model_name']))
res_str = json.dumps(res)
print('res..' + res_str[:200])
print('res info: ',res['info'])
print('res pred 1: ',res['Predictions'][0])

predict_json_output_file_name = predict_json_input_file_name.split('.')[0]
file_name_parts = predict_json_output_file_name.split('_')
if file_name_parts[-1] == 'input':
    file_name_parts[-1] = 'predictions'
else:
    file_name_parts.append('predictions')
predict_json_output_file_name = '_'.join(file_name_parts)  + '.json'
full_path_json_out = os.path.join(finder.get_output_folder(),predict_json_output_file_name)
print('writing prediction out: ',predict_json_output_file_name)
with open(full_path_json_out, 'w') as outfile:
    json.dump(res, outfile, indent=4)




