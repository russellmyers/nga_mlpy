'''
Script used to perform route pipeline step, either locally or within Azure
'''


from ngamlfpy.pipeline import FileRouter

import argparse
import os

print("In pipeline_step_route_data.py.....")

parser = argparse.ArgumentParser("pipeline_step_route_data")

parser.add_argument("--input_data", type=str,default='./data', help="input data directory")
#parser.add_argument("--output_routed", type=str, help="output_routed directory")  eg  data/TWV/input
#parser.add_argument("--pipeline_parameter", type=str, help="pipeline parameter test")
parser.add_argument("--ml_service",type=str,help="Machine learning service code, eg TWV")
#parser.add_argument("--workspace_name",type=str,help="Azure workspace name, eg twv_test-ws")

args = parser.parse_args()

print("Argument 1: %s" % args.input_data)
print("Argument 2: %s" % args.ml_service)
#print("Argument 3: %s" % args.workspace_name)


file_router = FileRouter(args.ml_service,base_folder=args.input_data) #os.path.join(args.input_data,'data'))

all_files = file_router.route_files()

print('Routing complete. ', len(all_files), ' routed')
print ('Pipeline complete')