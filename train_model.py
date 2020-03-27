'''

Training code script.

Can be used locally or run via Azure Experiment.

If run locally, can still opt to log results to Azure


'''


from ngamlfpy.train import AzureTrainer
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder',default='./data', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--regularization', type=float, dest='hp_reg', default=0.0001, help='regularization rate')
    parser.add_argument('--learning-rate', type=float, dest='hp_lr_init', default=0.001, help='initial learning rate')
    parser.add_argument('--iters' , type=int, dest='hp_iters', default=1000, help='iterations')
    parser.add_argument('--hidden-layers',nargs="*",type=int, dest='hp_hidden_layers', default=[100], help='hidden layers')
    parser.add_argument('--local-run-log-to-azure',type=str, dest='local_run_log_to_azure', default='N', help='local run with logging to azure')
    parser.add_argument('--local-log-experiment-name',type=str, dest='local_log_experiment_name', default='deep_neural_network', help='local run experiment name for logging to azure')
    parser.add_argument('--local-log-ws-config-path',type=str,dest='local_log_ws_config_path',default=None,help='local ws config file path')
    parser.add_argument('--model',type=str, dest='model', default='T001', help='Model code eg T001')
    parser.add_argument('--model-version', type=str, dest='model_version', default=None, help='Model  version to train, eg 002. If not specified, uses latest version')
    parser.add_argument('--ml_service',type=str, dest='ml_service', default='PAD', help='ML service eg TWV')
    parser.add_argument('--clip_training_set',type=int,default=-1,help='Clipped training set size')
    parser.add_argument('--clip_test_set', type=int, default=-1, help='Clipped test set size')
    args = parser.parse_args()

    return args

print('Training commencing....')

args = get_args()


trainer = AzureTrainer.trainer_from_args(args)


nn_model,avg_cost, avg_test_cost, acc_trn, acc_tst = trainer.train(trained_model_in=None,report_progress=True)

trainer.write_model()

trainer.plot_graphs()

trainer.run.complete()
print('Training is complete!')

