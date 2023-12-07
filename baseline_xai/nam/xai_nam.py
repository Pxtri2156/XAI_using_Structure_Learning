import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nam.config import defaults
from nam.data import FoldedDataset, NAMDataset
from nam.models import NAM, get_num_units
from nam.trainer import LitNAM
from nam.utils import *
import sklearn
import pandas as pd 
import os 
import argparse
import numpy as np
import json 
import random
# Read dataset

def main(args):

  # Set config
  config = defaults()
  print(config)
  config.device = 'cpu'
  config.num_epochs=args.epochs
  config.regression = args.regression
  config.logdir = args.root_path
  config.save_dir = f"{args.root_path}/save_dir"
  if not os.path.isdir(config.save_dir):
      os.mkdir(config.save_dir)
  config.name = f"NAM_{args.data_name}"
  config.version = "1.0"
  
  x1_list = []
  x2_list = []
  acc_list = []
  f1_list = []
  mse_list = []

  for seed in range(args.no_seeds):
    random.seed(seed)
    np.random.seed(seed)
    seed_name = 'seed_'+str(seed)
    # out_folder = args.root_path + f"{seed_name}"
    # if not os.path.isdir(out_folder):
    #     os.mkdir(out_folder)
    # out_folder += "/"
    dataset = pd.read_csv(args.data_path)
    dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset = NAMDataset(config,
                          data_path=dataset,
                          features_columns=dataset.columns[:-1],
                          targets_column=dataset.columns[-1])

    ## Getting the training dataloaders
    trainloader, valloader = dataset.train_dataloaders()

    model = NAM(
      config=config,
      name=config.name,
      num_inputs=len(dataset[0][0]),
      num_units=get_num_units(config, dataset.features),
    )

    tb_logger = TensorBoardLogger(config=config)

    checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                            monitor='val_loss',
                                            save_top_k=config.save_top_k,
                                            mode='min')

    litmodel = LitNAM(config, model)
    trainer = pl.Trainer(logger=tb_logger, max_epochs=config.num_epochs, callbacks=checkpoint_callback)
    trainer.fit(litmodel, train_dataloaders=trainloader, val_dataloaders=valloader)
    results = trainer.test(litmodel, dataloaders=valloader)
    print("results: ", results)
    if args.regression is False:
    # results = trainer.predict_step()
      acc = results[0]['Accuracy_metric_epoch']
      f1 = results[0]['f1_score_epoch']
      acc_list.append(acc)
      f1_list.append(f1)

    else:
      mse = results[0]["MAE_metric_epoch"]
      mse_list.append(mse)
      
    # fig = plot_mean_feature_importance(litmodel.model, dataset)
    # fig.savefig(f"{out_folder}NAM_{args.data_name}.png")    
    # fig.clf()
    # compute importance features 
    mean_pred, avg_hist_data = calc_mean_prediction(litmodel.model, dataset)
    def compute_mean_feature_importance(mean_pred, avg_hist_data):
        mean_abs_score = {}
        for k in avg_hist_data:
            try:
                mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
            except:
                continue
        x1, x2 = zip(*mean_abs_score.items())
        return x1, x2

    ## TODO: rename x1 and x2
    x1, x2 = compute_mean_feature_importance(mean_pred, avg_hist_data)
    x2_list.append(x2)
    
  # Calculate avg result 
  acc_list = np.array(acc_list)
  f1_list = np.array(f1_list)
  mse_list = np.array(mse_list)
  final_cls_results = { 'accuracy': {'mean': np.mean(acc_list), 
                                      'std': np.std(acc_list)},
                      'f1_score': {'mean': np.mean(f1_list),
                                  'std': np.std(f1_list)},
                      'mse': {'mean': np.mean(mse_list),
                              'std': np.std(mse_list)}
  }  
  json_object = json.dumps(final_cls_results, indent=4)
  # Writing to sample.json
  with open(args.root_path + "final_result.json", "w") as outfile:
      outfile.write(json_object)
  # Draw 
  x2_list = np.array(x2_list)
  x2_mean = np.mean(x2_list,axis=0)
  cols = dataset.features_names
  fig = plt.figure(figsize=(12,12))
  ind = np.arange(len(x1))
  x1_indices = np.argsort(x2_mean)

  cols_here = [cols[i] for i in x1_indices]
  x2_here = [x2_mean[i] for i in x1_indices]
  width=0.65
  plt.barh(ind, x2_here, width)
  plt.yticks(ind , cols_here, fontsize='large')
  plt.xticks(fontsize = 'large')
  plt.xlabel('Mean Absolute Score', fontsize="x-large")
  fig.savefig(f'{args.root_path}/NAM_avg_{args.data_name}.pdf')
  plt.clf()
  
def additive_arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--root_path", 
                        default="/workspace/tripx/MCS/xai_causality/nam_run/ov_cancer/", 
                        type=str)
    parser.add_argument("--data_path", 
                        default="/dataset/PANCAN/OV_gene_filter.csv", 
                        type=str)
    parser.add_argument("--data_name", 
                        default="ov_cancer", 
                        type=str)
    parser.add_argument("--regression", 
                        default=False, 
                        type=bool,
                        help='Number of epochs')
    parser.add_argument("--no_seeds", 
                        default=20, 
                        type=int,
                        help='Number of random seed')
    parser.add_argument("--epochs", 
                        default=100, 
                        type=int,
                        help='Number of epochs')
    # log
    parser.add_argument("--wandb_mode", 
                        default="disabled", 
                        type=str)
    
    return parser.parse_args()
  
if __name__ == '__main__':
  args = additive_arg_parser()
  main(args)
