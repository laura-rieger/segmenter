from functools import lru_cache
from typing import Tuple
import numpy as np



import argparse
import configparser
import os
import platform
from perqueue.task_classes.task_groups import _ToggleGroup
from os.path import join as oj

    
from tqdm import tqdm

import sys
from copy import deepcopy

import pickle as pkl
is_windows = platform.system() == "Windows"
num_workers = 0 if is_windows else 4


def main(foldername: str = '5570988367',  **kwargs) -> Tuple[bool, dict]:
    import torch
    import my_data
    from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
    from unet import UNet
    import evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    results = {}
    config = configparser.ConfigParser()

    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")
    tot_folder_path = oj(config["PATHS"]["pq_progress_results"], foldername)
    # load pkl file
    results = pkl.load(open(oj(tot_folder_path, "results.pkl"), "rb"))
    # load pool
    x_pool = my_data.load_pool_data( oj(config["DATASET"]["data_path"], results['poolname']) )[:-2]
    x_pool_all, slice_numbers = my_data.make_dataset_single( x_pool, 
                                                #  img_size=args.image_size*2,
                                                 img_size=results["image_size"],
                                                 offset=results["image_size"],
                                                  return_slice_numbers= True )
    del x_pool
        # x_pool_all = (x_pool_all - data_min) / (data_max - data_min)
    pool_set = TensorDataset(torch.from_numpy(x_pool_all))

  
    loader_args = dict( batch_size=results['batch_size'], num_workers=num_workers, pin_memory=True )
    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)

    net = UNet(n_channels=1, n_classes=results["num_classes"], ).to(device=device)
    net.load_state_dict(torch.load(oj(tot_folder_path, "model_state.pt")))
    cost_function = getattr(evaluate, results['cost_function'])



    add_ids = cost_function( net, device, pool_loader,  (results["data_min"], results["data_max"]), n_choose=results['add_size'],)
    num_val_add = np.maximum(int(len(add_ids) * results['val'] / 100), 1) if results['add_size'] >1 else 0

    add_train_ids = add_ids[:-num_val_add]
    add_val_ids = add_ids[-num_val_add:]
    add_indicator_list = [0 for _ in range(len(add_train_ids))] + [ 1 for _ in range(len(add_val_ids)) ]
    add_list = [x for x in add_ids]
    remove_id_list = pkl.load( open( oj( config["PATHS"]["pq_progress_results"], foldername, "image_idxs.pkl", ), "rb", ) )


    remove_id_list.append(add_list)
    cur_folder = oj(config["PATHS"]["pq_progress_results"], results['file_name'])


    my_data.make_check_folder(config["PATHS"]["pq_progress_results"], results['file_name'])


    torch.save(net.state_dict(), oj(cur_folder, "model_state.pt"))



    my_data.save_progress(net, 
                            remove_id_list, 
                            x_pool_all[add_ids,], #:,add_crop_start:add_crop_end,add_crop_start:add_crop_end],
                            config["PATHS"]["pq_progress_results"], 
            
                            device, 
                            results, results['class_dict'], add_indicator_list,slice_numbers )
    




    return (True, {'foldername':results["file_name"], 'pool_name': results['poolname'],'image_size': results['image_size'],_ToggleGroup.keyword():False})






if __name__ == '__main__':
    _, d = main(poolname = 'pq_lno')
    # print(d)

