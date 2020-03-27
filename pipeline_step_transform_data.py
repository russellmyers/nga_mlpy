'''
Script used to perform transform pipeline step, either locally or within Azure
'''

from ngamlfpy.pipeline import PipelineTransformer
import json

import argparse

print("In pipeline_step_transform_data.py.....")

parser = argparse.ArgumentParser("pipeline_step_transform_data")

parser.add_argument("--input_data",default='./data',type=str, help="input data directory")
#parser.add_argument("--output_transformed", type=str, help="output_transformed directory")  eg data/TWV/train
parser.add_argument("--pipeline_parameter", type=str,default="23", help="pipeline parameter test")
parser.add_argument("--ml_service",type=str,help="Machine learning service code, eg TWV")
parser.add_argument("--model_name",type=str,default="",help="Machine learning model name, eg T001")
parser.add_argument("--model_version",type=str,default="",help="Model version to transform - latest version if left blank")
parser.add_argument("--json_input",type=str,default="",help="json input to transform")
parser.add_argument("--normalise_labels",type=str,default="N",help="label normalisation flag eg Y/N")


args = parser.parse_args()

print("Argument 1: %s" % args.input_data)
#print("Argument 2: %s" % args.output_transformed)
print("Argument 2: %s" % args.pipeline_parameter)
print("Argument 3: %s" % args.ml_service)
print("Argument 4: %s" % args.model_name)
print("Argument 5: %s" % args.model_version)
print("Argument 6 length: %s" %  (str(len(args.json_input))))
print("Argument 7: %s"  % args.normalise_labels)

model = None
model_vers = None
in_json = None

if args.model_name != '':
    model =  args.model_name

if args.model_version != '':
    model_vers =  args.model_version

if args.normalise_labels == 'Y':
    normalise_labels=True
else:
    normalise_labels = False

if args.json_input == '' or args.json_input == '{}':
    pass
else:
    in_json = json.loads(args.json_input)
    print('in Json selection: %s' % in_json['selection'])
    print('in Json num emps: %s' % str(len(in_json['values'])))


transformer = PipelineTransformer(args.ml_service, base_folder = args.input_data,model_name=model,model_version = model_vers, in_json=in_json,normalise_labels=normalise_labels)
transformed_df = transformer.process_data()

print('Transform complete')

