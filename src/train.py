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


def main(poolname: int = 0, add_ratio: int = .02, needed_accuracy: float = .95, progress_folder: str = None,   **kwargs) -> Tuple[bool, dict]:
    import torch
    results = {}
    np.random.seed()


    
    
    args = get_args()
    for arg in vars(args):
        results[str(arg)] = getattr(args, arg)


    config = configparser.ConfigParser()

    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")
     
    if progress_folder != None:
        print("Continuing run")

        args.progress_folder =progress_folder
        run_folder = oj(config["PATHS"]["pq_progress_results"], str(args.progress_folder))
        args = pkl.load(open(oj(run_folder, "args.pkl"), "rb"))
        args.progress_folder = progress_folder


        results = pkl.load(open(oj(run_folder, "results.pkl"), "rb"))
    else:
        results["file_name"] = "".join([str(np.random.choice(10)) for x in range(10)])
        args.add_ratio = add_ratio
        args.poolname = poolname    
        results['al_progress'] = []




    save_path = config["PATHS"]["pq_model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.noGPUavailable() # if cuda is not there, fail loud af!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    best_score, _ = run( device=device, args=args, config =  config, results=results )

    exit_cond =  best_score >= needed_accuracy 



    return (True, {'foldername':results["file_name"], 'val_score': best_score, 'pool_name': args.poolname,'image_size': args.image_size,_ToggleGroup.keyword():exit_cond})


def train(net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, num_batches):

    
    import torch
    torch.backends.cudnn.deterministic = True
    from torch.nn import functional as F
    from utils.dice_score import dice_loss
    
    from unet import UNet

    net.train()
    epoch_loss = 0
    for i, ( images, true_masks, ) in enumerate(train_loader):  
        # we do augmentation here
        if np.random.uniform() > 0.5:
            images = torch.flip(images, [ 2, ], )
            true_masks = torch.flip( true_masks, [ 1, ], )
        if np.random.uniform() > 0.5:
            images = torch.flip( images, [ 3, ], )
            true_masks = torch.flip( true_masks, [ 2, ], )
        
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=True):
            if torch.any(true_masks != 255):
                masks_pred = net(images) #+ 1e-8 # stabilize
                # focal loss
                
                # try:
                masks_pred = F.softmax(masks_pred, dim=1)
                masks_pred = masks_pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
                true_masks = true_masks.view(-1)

                loss = criterion(masks_pred, true_masks)
                loss_dice = dice_loss( F.softmax(masks_pred, dim=1).float(), true_masks, num_classes, multiclass=True, )
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss + loss_dice).backward()
   
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += loss.item()  
                # epoch_loss += loss.item() 
        if i >= num_batches:
            break
    return epoch_loss / (num_batches )  


def run(device, args, config, results):
    save_path = config["PATHS"]["pq_model_path"]
    
    from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
    import evaluate
    from unet import UNet
    from torch import optim
    import my_data
    from focal_loss.focal_loss import FocalLoss
    import torch
    print("in run")
    sys_exit = False
    print("Start setting up data")
    loader_args = dict( batch_size=args.batch_size, num_workers=num_workers, pin_memory=True )
    cost_function = getattr(evaluate, args.cost_function)
    is_human_annotation = ( len(os.listdir(oj(config["DATASET"]["data_path"], args.poolname))) == 1 )
    x, y, num_classes, class_dict = my_data.load_layer_data( oj(config["DATASET"]["data_path"], args.foldername) )
    results["class_dict"] = class_dict
    data_min, data_max = np.min(x[:-1]), np.max(x[:-1])
    results["data_min"] = data_min
    results["data_max"] = data_max
    x = (x - data_min) / (data_max - data_min)
    results.setdefault("val_scores", [])
    results.setdefault("train_losses", [])
    results.setdefault("num_classes", num_classes)
    x, y = x[:-1], y[:-1]  # #  don't touch the last full image - left for test
    # if it s not human 
    all_idxs = np.arange(len(x))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_val = np.ceil(len(x) * args.val / 100).astype(int)
    n_train = len(x) - n_val
    all_train_idxs = all_idxs[:n_train]
    val_idxs = all_idxs[n_train:]
    
    init_train_idxs = all_train_idxs

    train_set = TensorDataset( *[ torch.Tensor(input) for input in my_data.make_dataset(x[init_train_idxs], y[init_train_idxs], img_size=args.image_size, offset=args.offset, ) ] )

    # pretend we have a validation set that is annotated
    x_val, y_val, _, _ = my_data.load_layer_data( oj(config["DATASET"]["data_path"], 'lno') )
    x_val, y_val = x_val[-2:-1], y_val[-2:-1]
    x_val = (x_val - data_min) / (data_max - data_min)


    val_set = TensorDataset( *[ torch.Tensor(input) for input in my_data.make_dataset(x_val, y_val, img_size=args.image_size, offset=args.offset, ) ] )
    new_val_set = None
    new_val_loader = None
    
    if is_human_annotation:
        x_pool = my_data.load_pool_data( oj(config["DATASET"]["data_path"], args.poolname) )[:-2]
        x_pool_all, slice_numbers = my_data.make_dataset_single( x_pool, 
                                                #  img_size=args.image_size*2,
                                                 img_size=args.image_size,
                                                 offset=args.image_size,
                                                  return_slice_numbers= True )
        del x_pool
        # x_pool_all = (x_pool_all - data_min) / (data_max - data_min)
        pool_set = TensorDataset(torch.from_numpy(x_pool_all))

    else:
        x_pool, y_pool, _, _ = my_data.load_layer_data( oj(config["DATASET"]["data_path"], args.poolname) )
  
        x_pool, y_pool = x_pool[:-2], y_pool[:-2]

        x_pool_all, y_pool_all = my_data.make_dataset( x_pool, y_pool, img_size=args.image_size, offset=args.image_size, )
        pool_set = TensorDataset( *[ torch.from_numpy(x_pool_all), torch.from_numpy(y_pool_all), ] )
    initial_pool_len = len(pool_set)

    weight_factor = 10

    weights = [1 for _ in range(len(train_set))]
    new_weights = weights
    # if this is a continuation, load the data
    remove_id_list = []
    if args.progress_folder != "" and is_human_annotation:
        results["file_name"] = args.progress_folder
        pool_ids = np.arange(len(x_pool_all))
        remove_id_list = pkl.load( open( oj( config["PATHS"]["pq_progress_results"], args.progress_folder, "image_idxs.pkl", ), "rb", ) )
        for remove_ids in remove_id_list:
            cur_remove_list = [x for x in remove_ids] # useless
            pool_ids = np.delete(pool_ids, cur_remove_list, axis=0)
            slice_numbers = np.delete(slice_numbers, cur_remove_list, axis=0)
        x_pool_all = x_pool_all[pool_ids]
        pool_set = TensorDataset( *[ torch.from_numpy(x_pool_all), ]  ) # dtype = torch.uint8
        (train_add_set, new_val_set) = my_data.load_annotated_imgs( oj( config["PATHS"]["pq_progress_results"], args.progress_folder, ), class_dict, (data_min, data_max) )
        new_val_loader = DataLoader(new_val_set, shuffle=False, **loader_args)
        weights = [1 for _ in range(len(train_set))] + [ weight_factor for _ in range(len(train_add_set)) ]
        train_set = ConcatDataset([train_set, train_add_set])
        

    torch.manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    print("Start setting up model")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = UNet(n_channels=1, n_classes=results["num_classes"], ).to(device=device)
    if args.progress_folder != "":
    
        
        # net.load_state_dict(torch.load(oj(run_folder, "model_state.pt")))
        pass

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, )
    grad_scaler = torch.cuda.amp.GradScaler()

    criterion = FocalLoss(gamma=0.7, ignore_index=255)
    best_val_score = 0
    best_weights = None
    patience = args.final_patience if args.add_step == 0 else args.add_step
    cur_patience = 0
    print("Start training")
    # tqdm total is patience if add step is unequal zero, otherwise args.epoch



    tqdm_total = patience if args.add_step != 0 and is_human_annotation else args.epochs
    init_epochs = len(
        results["val_scores"]
    )  # if this is in progress, we start at the current epoch
    # init_epochs = 0 # 
    for epoch in tqdm(range(init_epochs, args.epochs + 1), total=tqdm_total):
        train_loss = train(net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, args.num_batches)
        val_score = evaluate.evaluate(net, val_loader, device, num_classes, criterion)
        old_val_score = val_score
        if new_val_set is not None:
            new_val_score = evaluate.evaluate(net, new_val_loader, device, num_classes, criterion)
            # if the new val set is not none, we need to weigh the scores
            val_score = ( val_score * len(val_loader.dataset) + new_val_score * len(new_val_loader.dataset) * weight_factor ) / (len(val_loader.dataset) + len(new_val_loader.dataset) * weight_factor)
        results["val_scores"].append(val_score)
        results["train_losses"].append(train_loss)
        # print(val_score, train_loss, new_val_score if new_val_set is not None else 0, old_val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_weights = deepcopy(net.state_dict())
            cur_patience = 0
        else:
            cur_patience += 1
        # print(args.epochs)
        if cur_patience > patience or epoch == args.epochs:   
        
   
        
            net.eval()
            if ( len(pool_loader.dataset) > 0 and len(pool_loader.dataset) / initial_pool_len > 1 - args.add_ratio ):
                cur_patience = 0

                net.load_state_dict(best_weights) 
                if not os.path.exists(config["PATHS"]["pq_progress_results"]):
                    os.makedirs(config["PATHS"]["pq_progress_results"])    
                add_ids = cost_function( net, device, pool_loader,  (data_min, data_max), n_choose=args.add_size,)
                num_val_add = np.maximum(int(len(add_ids) * args.val / 100), 1) if args.add_size >1 else 0

                add_train_ids = add_ids[:-num_val_add]
                add_val_ids = add_ids[-num_val_add:]
                add_indicator_list = [0 for _ in range(len(add_train_ids))] + [ 1 for _ in range(len(add_val_ids)) ]
                add_list = [x for x in add_ids]


                remove_id_list.append(add_list)
                cur_folder = oj(config["PATHS"]["pq_progress_results"], results['file_name'])


                my_data.make_check_folder(config["PATHS"]["pq_progress_results"], results['file_name'])
          

                torch.save(net.state_dict(), oj(cur_folder, "model_state.pt"))
                pkl.dump(args, open(oj(cur_folder, "args.pkl"), "wb"))
    


                my_data.save_progress(net, 
                                        remove_id_list, 
                                        x_pool_all[add_ids,], #:,add_crop_start:add_crop_end,add_crop_start:add_crop_end],
                                        config["PATHS"]["pq_progress_results"], 
                                        args, 
                                        device, 
                                        results, class_dict, add_indicator_list,slice_numbers )
                print(results["file_name"])
                sys_exit = True

            else:
                # if the pool is empty, increase patience to ten, a normal patience value and continue training with the full set
                if patience != args.final_patience:
                    patience = args.final_patience
                    print("Patience increased to {} for final training".format(patience))
                else:
                    net.load_state_dict(best_weights)
                    net.eval()

                    results["final_dice_score"] = evaluate.evaluate( net, val_loader, device, 
                                                                    num_classes, criterion)
              
                    #load data again
                    if "lno" in args.foldername.lower():
                        x, y, num_classes, class_dict = my_data.load_layer_data( oj(config["DATASET"]["data_path"], 'lno') )
                        
                        x_test, y_test = x[-1:], y[-1:]
                        # x_test, y_test = my_data.stack_imgs(x_test, y_test)
                        x_test = (x_test - data_min) / (data_max - data_min)

                        results["test_dice_score"] = evaluate.final_evaluate(net, x_test, y_test, 
                                                                            num_classes, device)
                        all_idxs = np.arange(len(x))
                        np.random.seed(0)
                        np.random.shuffle(all_idxs)
                        n_val = np.ceil(len(x) * args.val / 100).astype(int)
                        n_train = len(x) - n_val
                        val_idxs = all_idxs[n_train:]
                        
                        # x_val, y_val = my_data.stack_imgs(x[val_idxs], y[val_idxs])
                        x_val = (x[val_idxs] - data_min) / (data_max - data_min)
                        y_val = y[val_idxs]
                        results["final_dice_score"] = evaluate.final_evaluate(net, x_val,y_val,
                                                                            num_classes, device)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    results['al_progress'].append( best_val_score )
                    print(results['al_progress'])
                    pkl.dump(results, open(os.path.join(save_path, results["file_name"] + ".pkl"), "wb") )
                    
                    torch.save(net.state_dict(), oj(save_path, results["file_name"] + ".pt"))
        
                    sys_exit = True
        if sys_exit:
            break
    return best_val_score, len(pool_loader.dataset) / initial_pool_len > (1 - args.add_ratio )
            


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )

    parser.add_argument( "--learningrate", "-l", type=float, default=0.0001, dest="lr", )
    parser.add_argument( "--image-size", dest="image_size", type=int, default=128, ) 
    parser.add_argument("--epochs", "-e", type=int, default=5000)
    parser.add_argument( "--batch-size", "-b", dest="batch_size", type=int, default=128, )
    parser.add_argument( "--cost_function", dest="cost_function", type=str, default="cut_off_cost", )
    parser.add_argument( "--add_ratio", type=float, default=0.02, ) # increase for actual experim
    

    parser.add_argument( "--poolname", type=str, default="lno", )
    parser.add_argument( "--experiment_name", "-g", type=str, default="", )

    parser.add_argument( "--add_size", type=int, help="How many patches should be added to the training set in each round", default=4, )
    parser.add_argument( "--offset", dest="offset", type=int, default=64, )
    parser.add_argument( "--seed", "-t", type=int, default=42, )
    parser.add_argument( "--validation", "-v", dest="val", type=int, default=18, help="Val percentage (0-100)", )
    parser.add_argument( "--export_results", type=int, default=0, help="If the added samples should be exported - this is for presentation slides", )
    parser.add_argument( "--add_step", type=int, default=0, help="> 0: examples will be added at preset intervals rather than considering the validation loss when validation set is sparsely annotated", )
    parser.add_argument("--progress_folder", "-f", type=str, default="") 
    parser.add_argument("--final_patience", "-fp", type=int, default="5")
    
    parser.add_argument("--num_batches", "-nb", type=int, default="64")

    parser.add_argument( "--foldername", type=str, default="lno_halfHour", )
    return parser.parse_args()








if __name__ == '__main__':
    _, d = main(poolname = 'pq_lno')
    # print(d)

