import argparse
import configparser
import os
import platform
import my_data
from os.path import join as oj
import torch
torch.backends.cudnn.deterministic = True
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from tqdm import tqdm
from focal_loss.focal_loss import FocalLoss
import sys
from copy import deepcopy
from torch import optim
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from torch.nn import functional as F
from utils.dice_score import dice_loss
import evaluate
from unet import UNet

is_windows = platform.system() == "Windows"
num_workers = 0 if is_windows else 4
results = {}
np.random.seed()


results["file_name"] = "".join([str(np.random.choice(10)) for x in range(10)])

def train(net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, num_batches):
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
        # images = images + torch.randn(images.shape) * .01

        

        
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=True):
            if torch.any(true_masks != 255):
                masks_pred = net(images)
                # focal loss
                try:
                    masks_pred = F.softmax(masks_pred, dim=1)
                    masks_pred = masks_pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
                    true_masks = true_masks.view(-1)
                
                    loss = criterion(masks_pred, true_masks)
                    loss_dice = dice_loss( F.softmax(masks_pred, dim=1).float(), true_masks, num_classes, multiclass=True, )
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss + loss_dice).backward()
                except:
                    print("Error in loss calculation")
                    print(masks_pred.shape)
                    print(true_masks.shape)
                    print(masks_pred.max(), masks_pred.min())
                    print(true_masks.max(), true_masks.min())
                    sys.exit()
               
                # grad_scaler.scale(loss ).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += loss.item() + loss_dice.item()
                # epoch_loss += loss.item() 
        if i >= num_batches:
            break
    return epoch_loss / (num_batches )  


def run(device, args):
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
    val_set = TensorDataset( *[ torch.Tensor(input) for input in my_data.make_dataset(x[val_idxs], y[val_idxs], img_size=args.image_size, offset=args.offset, ) ] )
    new_val_set = None
    new_val_loader = None
    
    if is_human_annotation:
        x_pool = my_data.load_pool_data( oj(config["DATASET"]["data_path"], args.poolname) )
        x_pool_all, slice_numbers = my_data.make_dataset_single( x_pool, 
                                                #  img_size=args.image_size*2,
                                                 img_size=args.image_size,
                                                 offset=args.image_size,
                                                  return_slice_numbers= True )
        del x_pool
        x_pool_all = (x_pool_all - data_min) / (data_max - data_min)
        pool_set = TensorDataset(torch.from_numpy(x_pool_all))

    else:
        x_pool, y_pool, _, _ = my_data.load_layer_data( oj(config["DATASET"]["data_path"], args.poolname) )
  
        x_pool, y_pool = x_pool[:-1], y_pool[:-1]
  
        x_pool_all, y_pool_all = my_data.make_dataset( x_pool, y_pool, img_size=args.image_size, offset=args.image_size, )
        pool_set = TensorDataset( *[ torch.from_numpy(x_pool_all), torch.from_numpy(y_pool_all), ] )
    initial_pool_len = len(pool_set)

    weight_factor = 5

    weights = [1 for _ in range(len(train_set))]
    new_weights = weights
    # if this is a continuation, load the data
    remove_id_list = []
    if args.progress_folder != "" and is_human_annotation:
        results["file_name"] = args.progress_folder
        pool_ids = np.arange(len(x_pool_all))
        remove_id_list = pkl.load( open( oj( config["PATHS"]["progress_results"], args.progress_folder, "image_idxs.pkl", ), "rb", ) )
        for remove_ids in remove_id_list:
            cur_remove_list = [x for x in remove_ids] # useless
            pool_ids = np.delete(pool_ids, cur_remove_list, axis=0)
            slice_numbers = np.delete(slice_numbers, cur_remove_list, axis=0)
        x_pool_all = x_pool_all[pool_ids]
        pool_set = TensorDataset( *[ torch.from_numpy(x_pool_all), ]  ) # dtype = torch.uint8
        (train_add_set, new_val_set) = my_data.load_annotated_imgs( oj( config["PATHS"]["progress_results"], args.progress_folder, ), class_dict, )
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
        net.load_state_dict(torch.load(oj(run_folder, "model_state.pt")))

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, )
    grad_scaler = torch.cuda.amp.GradScaler()
    # criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean') 
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
    for epoch in tqdm(range(init_epochs, args.epochs + 1), total=tqdm_total):
        train_loss = train(net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, args.num_batches)
        val_score = evaluate.evaluate(net, val_loader, device, num_classes, criterion)
        if new_val_set is not None:
            new_val_score = evaluate.evaluate(net, new_val_loader, device, num_classes, criterion)
            # if the new val set is not none, we need to weigh the scores
            val_score = ( val_score * len(val_loader.dataset) + new_val_score * len(new_val_loader.dataset) * weight_factor ) / (len(val_loader.dataset) + len(new_val_loader.dataset) * weight_factor)
        results["val_scores"].append(val_score)
        results["train_losses"].append(train_loss)

        if val_score > best_val_score:
            best_val_score = val_score
            best_weights = deepcopy(net.state_dict())
            cur_patience = 0
        else:
            cur_patience += 1

        if cur_patience > patience or epoch == args.epochs: 
        # if True:
        
            net.eval()
            if ( len(pool_loader.dataset) > 0 and len(pool_loader.dataset) / initial_pool_len > 1 - args.add_ratio ):
                cur_patience = 0
                add_ids = cost_function( net, device, pool_loader,  (data_min, data_max), n_choose=args.add_size,)
                num_val_add = np.maximum(int(len(add_ids) * args.val / 100), 1) if args.add_size >1 else 0

                add_train_ids = add_ids[:-num_val_add]
                add_val_ids = add_ids[-num_val_add:]
                add_indicator_list = [0 for _ in range(len(add_train_ids))] + [ 1 for _ in range(len(add_val_ids)) ]
                add_list = [x for x in add_ids]

                if is_human_annotation:
                    if not os.path.exists(config["PATHS"]["progress_results"]):
                        os.makedirs(config["PATHS"]["progress_results"])    
                    remove_id_list.append(add_list)
                    net.load_state_dict(best_weights) 

                    my_data.save_progress(net, 
                                          remove_id_list, 
                                          x_pool_all[add_ids,], #:,add_crop_start:add_crop_end,add_crop_start:add_crop_end],
                                          config["PATHS"]["progress_results"], 
                                          args, 
                                          device, 
                                          results, class_dict, add_indicator_list,slice_numbers )
                    print(results["file_name"])
                    sys.exit()
                else:

                    add_train_set = TensorDataset( *[ torch.Tensor((x_pool_all[add_train_ids,] - data_min) / (data_max - data_min)), 
                                                     torch.Tensor(y_pool_all[add_train_ids,]), ] )
                    add_val_set = TensorDataset( *[ torch.Tensor((x_pool_all[add_val_ids,] - data_min) / (data_max - data_min)), 
                                                   torch.Tensor(y_pool_all[add_val_ids,]), ] )
                    newTrainSet = ConcatDataset([train_loader.dataset, add_train_set])
      
                    if new_val_set is None:
                        new_val_set = add_val_set
                    else:
                        new_val_set = ConcatDataset([new_val_set, add_val_set])
                        
                    new_val_loader = DataLoader( new_val_set, shuffle=False, **loader_args )
                    # weigh the samples such as the total weight of them will be equal to the dataset
                    new_weights = new_weights  + [ weight_factor for _ in range(len(add_train_set)) ] 
                    new_sampler = torch.utils.data.WeightedRandomSampler( new_weights, len(new_weights) )
                    train_loader = DataLoader( newTrainSet, sampler=new_sampler, **loader_args )
                    # delete from pool
                    x_pool_all = np.delete(x_pool_all, add_ids, axis=0)
                    y_pool_all = np.delete(y_pool_all, add_ids, axis=0)
                    pool_set = TensorDataset( *[ torch.Tensor(x_pool_all), torch.Tensor(y_pool_all), ] )
                    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
                    best_val_score = 0
                    best_weights = None
                    print( "Added {} samples to the training set".format( len(add_train_ids) ) )
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
                    #xxxx not for graphite
                    #load data again
                    if 'lno' in args.foldername.lower():
                        x, y, num_classes, class_dict = my_data.load_layer_data( oj(config["DATASET"]["data_path"], 'lno') )
                        
                        x_test, y_test = x[-1:], y[-1:]
                        x_test, y_test = my_data.stack_imgs(x_test, y_test)
                        x_test = (x_test - data_min) / (data_max - data_min)

                        results["test_dice_score"] = evaluate.final_evaluate(net, x_test, y_test, 
                                                                            num_classes, device)
                        all_idxs = np.arange(len(x))
                        np.random.seed(0)
                        np.random.shuffle(all_idxs)
                        n_val = np.ceil(len(x) * args.val / 100).astype(int)
                        n_train = len(x) - n_val
                        val_idxs = all_idxs[n_train:]
                        
                        x_val, y_val = my_data.stack_imgs(x[val_idxs], y[val_idxs])
                        x_val = (x_val - data_min) / (data_max - data_min)
                        results["final_dice_score"] = evaluate.final_evaluate(net, x_val, y_val,
                                                                            num_classes, device)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    pkl.dump(results, open(os.path.join(save_path, results["file_name"] + ".pkl"), "wb") )
                    
                    torch.save(net.state_dict(), oj(save_path, results["file_name"] + ".pt"))
                    print(args.cost_function, results["file_name"]  )
                    sys.exit()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument("--epochs", "-e", type=int, default=2000)
    parser.add_argument( "--batch-size", "-b", dest="batch_size", type=int, default=2, )
    parser.add_argument( "--cost_function", dest="cost_function", type=str, default="cut_off_cost", )
    parser.add_argument( "--add_ratio", type=float, default=0.02, )
    parser.add_argument( "--foldername", type=str, default="lno_halfHour", )

    parser.add_argument( "--poolname", type=str, default="lno", )
    parser.add_argument( "--experiment_name", "-g", type=str, default="", )
    parser.add_argument( "--learningrate", "-l", type=float, default=0.001, dest="lr", )
    parser.add_argument( "--image-size", dest="image_size", type=int, default=128, )
    parser.add_argument( "--add_size", type=int, help="How many patches should be added to the training set in each round", default=2, )
    parser.add_argument( "--offset", dest="offset", type=int, default=64, )
    parser.add_argument( "--seed", "-t", type=int, default=42, )
    parser.add_argument( "--validation", "-v", dest="val", type=int, default=18, help="Val percentage (0-100)", )
    parser.add_argument( "--export_results", type=int, default=0, help="If the added samples should be exported - this is for presentation slides", )
    parser.add_argument( "--add_step", type=int, default=0, help="> 0: examples will be added at preset intervals rather than considering the validation loss when validation set is sparsely annotated", )
    parser.add_argument("--progress_folder", "-f", type=str, default="") 
    parser.add_argument("--final_patience", "-fp", type=int, default="5")
    
    parser.add_argument("--num_batches", "-nb", type=int, default="64")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = configparser.ConfigParser()
    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")
     
    if args.progress_folder != "":
        print("Continuing run")
        progress_folder = args.progress_folder

        run_folder = oj(config["PATHS"]["progress_results"], str(args.progress_folder))
        args = pkl.load(open(oj(run_folder, "args.pkl"), "rb"))
        args.progress_folder = progress_folder


        results = pkl.load(open(oj(run_folder, "results.pkl"), "rb"))

    for arg in vars(args):
        results[str(arg)] = getattr(args, arg)

    save_path = config["PATHS"]["model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.noGPUavailable() # if cuda is not there, fail loud af!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # if the run is continued, run_id is not empty
    run( device=device, args=args, )