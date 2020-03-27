import json
import pandas as pd
from ngamlfpy.utils import pulled_df_to_json
from ngamlfpy.hrxmlconfig import MLModelConfig
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str, dest='model_code', default='T001', help='Model code eg T001')
    args = parser.parse_args()

    return args

print('Conversion commencing....')

args = get_args()


if args.model_code == 'M010':
    model_version = '002'
    ml_service = 'PAD'
    payroll_service = 'EUH'
    gcc = 'AKN'
    lcc = 'A05'
    variant = 'ALL'
    system = 'EP5'
    client = '984'
    abkrs = 'N0'
    period = '201908'
elif args.model_code == 'M005':
    model_version = '001'
    ml_service = 'PAD'
    payroll_service = 'EUH'
    gcc = 'ZZZ'
    lcc = 'Z10'
    variant = 'MTHLY'
    system = 'EDF'
    client = '310'
    abkrs = 'ZZ'
    period = '201905'
elif args.model_code == 'A001':
    model_version = '001'
    ml_service = 'PAD'
    payroll_service = 'WDY'
    gcc = 'ALT'
    lcc = 'TST'
    variant = 'ALTBM'
    system = 'WDT'
    client = '000'
    abkrs = 'sm'
    period = '202003'
elif args.model_code == 'SS01':
    model_version = '011'
    ml_service = 'TWV'
    payroll_service = 'EUH'
    gcc = 'SOL'
    lcc = 'SUS'
    variant = 'BIWK'
    system = 'PP9'
    client = '100'
    abkrs = 'U3'
    period = '201915'

use_csv_input = True

save_csv_file_as_json = True

use_training_set_as_input = False

if use_training_set_as_input:
    input_csv_file_name = 'data/' + ml_service + '/train/' + args.model_code + '/' + model_version + '/' + args.model_code + '_train_master.csv'
else:
    input_csv_file_name = 'data/' + ml_service + '/input/' + args.model_code + '/' + model_version + '/' + '_'.join(
        [ml_service, args.model_code, model_version, payroll_service, gcc, lcc, variant, system, client, abkrs,
         period]) + '_input.csv'
output_json_folder =  'data/predict_test/' + args.model_code
os.makedirs(output_json_folder, exist_ok=True)
input_json_file_name = os.path.join(output_json_folder, args.model_code + '_test.json')
print(input_csv_file_name)
print(input_json_file_name)

df = pd.read_csv(input_csv_file_name)
#df = pd.read_csv('data/PAD/input/' + model_code + '/PAD_EUH_AKN_A05_ALL_EP5_984_N0_201908_input.csv')
#df = pd.read_csv('data/PAD/input/M005/001/PAD_M005_001_EUH_ZZZ_Z10_MTHLY_EDF_310_ZZ_201905_input.csv')
print(df.head())

ml_config = MLModelConfig.get_model_config_from_web_service_for_cust(ml_service, system=system,gcc=gcc,lcc=lcc,payroll_area=abkrs)

j_predict = pulled_df_to_json(df, ml_config, period, use_first_data_line_as_selection=True, use_value_title_format=True,
                              values_only=False, clip_emps=10)

outJson = input_json_file_name
print('Writing out json file: ', outJson)
with open(outJson, 'w') as outfile:
    json.dump(j_predict, outfile, indent=4)

predict_source = 'json'
# model_prefix = predict_file_folder
# predict_file_path = '/'.join([data_dir,model_prefix])
# full_predict_file_name = '/'.join([predict_file_path,predict_file_name])

print('Using Json predict file: ', input_json_file_name)

with open(input_json_file_name) as json_data:
    j_predict_verify = json.load(json_data)

# j_data_vals_stripped = []
# period = j_data['selection']['period']
# for val in j_data['values']:
#     if period == val['ForPeriod']:
#         j_data_vals_stripped.append(val)
# j_data['values']  = j_data_vals_stripped

print('Num emps: ', len(j_predict_verify['values']))
print('')
print('First emp: ', j_predict_verify['values'][0])
print('')
print('Selection: ', j_predict_verify['selection'])
print('')