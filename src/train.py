import argparse
import configparser
import os
import platform
import my_data
from os.path import join as oj
import torch
import torch.nn as nn
from tqdm import tqdm
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
file_name = "".join([str(np.random.choice(10)) for x in range(10)])
results["file_name"] = file_name

def train(
    net, train_loader, criterion, num_classes, optimizer, device, grad_scaler
):
    net.train()
    epoch_loss = 0
    num_batches = 64
    for i, (
        images,
        true_masks,
    ) in enumerate(train_loader):
        if np.random.uniform() > 0.5:
            images = torch.flip(images, [2, ],) 
            true_masks = torch.flip(true_masks, [1, ], )
        if np.random.uniform() > 0.5:
            images = torch.flip( images, [3, ], )
            true_masks = torch.flip( true_masks, [2, ], )
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=True):
            if torch.any(true_masks != 255):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks)
                loss_dice = dice_loss( F.softmax(masks_pred, dim=1).float(), true_masks, num_classes, multiclass=True, )
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss + loss_dice).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += loss.item()
        if i >= num_batches:
            break
    return epoch_loss / (num_batches * train_loader.batch_size)  # len(train_loader)


def train_net(device, args):
    print("Start setting up data")
    loader_args = dict( batch_size=args.batch_size, num_workers=num_workers, pin_memory=True )
    cost_function = getattr(evaluate, args.cost_function)
    results["val_scores"] = []
    results["train_losses"] = []
    # 1. Create dataset
    x, y, num_classes, class_dict = my_data.load_layer_data( oj(config["DATASET"]["data_path"], args.foldername))
    data_min, data_max = np.min(x[:-4]), np.max(x[:-4])
    x = (x - data_min) / (data_max - data_min)
    results["num_classes"] = num_classes
    x, y = x[:-4], y[:-4]  # #  don't touch the last full image - left for test

    all_idxs = np.arange(len(x))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_val = np.ceil(len(x) * args.val / 100).astype(int)
    n_train = len(x) - n_val
    all_train_idxs = all_idxs[:n_train]
    val_idxs = all_idxs[n_train:]
    init_train_idxs = all_train_idxs
    train_set = TensorDataset(*[torch.Tensor(input) for input in my_data.make_dataset(x[init_train_idxs], y[init_train_idxs], img_size=args.image_size, offset=args.offset,)])
    val_set = TensorDataset(*[torch.Tensor(input) for input in my_data.make_dataset(x[val_idxs], y[val_idxs], img_size=args.image_size, offset=args.image_size,)])
    new_val_set = None
    new_val_loader = None
    # num_train = len(train_set)
    is_human_annotation = (len(os.listdir(oj(config["DATASET"]["data_path"], args.poolname))) == 1 )
    if is_human_annotation:
        x_pool = my_data.load_pool_data(oj(config["DATASET"]["data_path"], args.poolname) )
        x_pool = (x_pool.astype(np.float16) - data_min) / (data_max - data_min)
        x_pool_all = my_data.make_dataset_single(x_pool, img_size=args.image_size, offset=args.image_size, )[0]
        pool_set = TensorDataset(torch.Tensor(x_pool_all))

    else:
        x_pool, y_pool, _, _ = my_data.load_layer_data( oj(config["DATASET"]["data_path"], args.poolname) )
        x_pool = (x_pool - data_min) / (data_max - data_min)
        x_pool, y_pool = x_pool[:-4], y_pool[:-4]
        x_pool_all, y_pool_all = my_data.make_dataset( x_pool, y_pool, img_size=args.image_size, offset=args.image_size, )
        pool_set = TensorDataset(*[torch.Tensor(x_pool_all), torch.Tensor(y_pool_all), ] )

    # 3. Create data loaders
    # the total weight of the added data should be equal to the training set
    weight_factor = len(train_set) / (len(pool_set)* args.add_ratio)

    weights = [1 for x in range(len(train_set))]
    new_weights = weights

    # Create a WeightedRandomSampler using the weights
    torch.manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
    initial_pool_len = len(pool_loader.dataset)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    # xxx needs to be changed before production
    if os.path.exists(oj(config["DATASET"]["data_path"], "lno")) and ("lno" in args.foldername or "LNO" in args.foldername):
        # load the fully annotated data to get the final evaluation on unseen "real" data
        x_final, y_final, _, _ = my_data.load_layer_data( oj(config["DATASET"]["data_path"], "lno") )
        x_final, y_final = x_final[:-4], y_final[:-4]  # just don't touch the last four
        x_final = (x_final - data_min) / (data_max - data_min)
        all_idxs_final = np.arange(len(x_final))
        np.random.seed(0)
        np.random.shuffle(all_idxs_final)
        n_val_final = np.maximum(int(len(x_final) * args.val / 100), 1)
        val_idxs_final = all_idxs_final[-n_val_final:]
        val_set_final = TensorDataset(*[torch.Tensor(input) for input in my_data.make_dataset( x_final[val_idxs_final], y_final[val_idxs_final], img_size=args.image_size, offset=args.image_size,)])
        final_val_loader = DataLoader( val_set_final, shuffle=False, drop_last=False, **loader_args)
    else:
        final_val_loader = val_loader
    torch.manual_seed(args.seed)
    print("Start setting up model")
    net = UNet(
        n_channels=1, n_classes=results["num_classes"], 
    ).to(device=device)
    optimizer = optim.Adam( net.parameters(), lr=args.lr,
    )
    grad_scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # 5. Begin training
    best_val_score = 0
    patience = 3 if args.add_step == 0 else args.add_step
    patience_delta = 0.005
    #if adding samples, assume initial dataset is roughly annotated and just add samples every fixed number of steps
    # todo add examples to the validation loss each time
    if args.add_step != 0: 
        patience = args.add_step
    # print out patience
    print("Patience is: " + str(patience))
    cur_patience = 0
    best_weights = None
    epoch = 0
    print("Start training")
    # tqdm total is patience if add step is unequal zero, otherwise args.epoch

    tqdm_total = patience if args.add_step != 0 and is_human_annotation else args.epochs
    for epoch in tqdm(range(1, args.epochs + 1), total=tqdm_total):
        train_loss = train( net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, )
        val_score = evaluate.evaluate(net, val_loader, device, num_classes).item()
        if new_val_set != None:
            new_val_score = evaluate.evaluate(net, new_val_loader, device, num_classes).item()
            print(new_val_score, val_score)
            print(len(new_val_loader.dataset) * weight_factor, len(val_loader.dataset))
            # obtain middle between old and new score
            val_score = (val_score * len(val_loader.dataset) + new_val_score * len(new_val_loader.dataset) * weight_factor) / (len(val_loader.dataset) + len(new_val_loader.dataset) * weight_factor)

        results["val_scores"].append(val_score)
        # print length of val scores
        print(results["val_scores"][-1])
        results["train_losses"].append(train_loss)

        
        # if the add step is unequal zero, just count up and add samples every add step
        if val_score > best_val_score + patience_delta:
            best_val_score = val_score
            best_weights = deepcopy(net.state_dict())
            cur_patience = 0
        else:
            cur_patience += 1

        if cur_patience > patience or epoch == args.epochs:

            print("Ran out of patience, ")
  
            net.eval()

            if ( len(pool_loader.dataset) > 0 and len(pool_loader.dataset) / initial_pool_len > 1 - args.add_ratio):
                cur_patience = 0
                add_ids = cost_function(net, device, pool_loader, n_choose=args.add_size)

                if is_human_annotation: 
                    #if human annotation, wait here for further
                    # check if this folder exists config["PATHS"]["progress_results"]
                    if not os.path.exists(config["PATHS"]["progress_results"]):
                        os.makedirs(config["PATHS"]["progress_results"])
                    my_data.save_progress(net, [add_ids, ], x_pool_all[add_ids], config["PATHS"]["progress_results"], file_name, args, device, results, class_dict, )
                    print(file_name)
                    sys.exit()
                else:
                    num_val_add = np.maximum(int(len(add_ids) * args.val / 100), 1)
                    add_train_ids = add_ids[:-num_val_add]
                    add_val_ids = add_ids[-num_val_add:]

                    add_train_set = TensorDataset(*[torch.Tensor(x_pool_all[add_train_ids]), torch.Tensor(y_pool_all[add_train_ids]),])
                    add_val_set = TensorDataset(*[torch.Tensor(x_pool_all[add_val_ids]), torch.Tensor(y_pool_all[add_val_ids]),])



                    # write out the new samples, the predictions and the labels for a presentation
                    newTrainSet = ConcatDataset([train_loader.dataset, add_train_set])
                    if new_val_set is None:
                        newValSet = ConcatDataset([val_loader.dataset, add_val_set])
                        new_val_loader = DataLoader(newValSet, shuffle=False, **loader_args)
                    else:
                        newValSet = ConcatDataset([new_val_set, add_val_set])

                    # weigh the samples such as the total weight of them will be equal to the dataset
                    new_weights = new_weights + [weight_factor for _ in range(len(add_train_set))]
                    new_sampler = torch.utils.data.WeightedRandomSampler(
                        new_weights, len(new_weights)
                    )
                    train_loader = DataLoader(
                        newTrainSet, sampler=new_sampler, **loader_args
                    )
                   
                    # delete from pool
                    x_pool_all = np.delete(x_pool_all, add_ids, axis=0)
                    y_pool_all = np.delete(y_pool_all, add_ids, axis=0)
                    pool_set = TensorDataset( *[ torch.Tensor(x_pool_all), torch.Tensor(y_pool_all), ] )
                    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
                    best_val_score = 0
                    print("Added {} samples to the training set".format(len(add_train_ids)))
            else:
                net.load_state_dict(best_weights)
               
                results["final_dice_score"] = evaluate.evaluate(net, final_val_loader, device, num_classes).item()
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb") )
                torch.save(net.state_dict(), oj(save_path, file_name + ".pt"))
                sys.exit()



def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument( "--batch-size", "-b", dest="batch_size", type=int, default=2, )
    parser.add_argument( "--cost_function", dest="cost_function", type=str, default="uncertainty_cost", )
    parser.add_argument( "--add_ratio", type=float, default=0.02, )
    parser.add_argument( "--foldername", type=str, default="lno_halfHour", )

    parser.add_argument( "--poolname", type=str, default="lno_full2", )
    parser.add_argument( "--experiment_name", "-g", type=str, default="", )
    parser.add_argument( "--learningrate", "-l", type=float, default=0.001, dest="lr", )
    parser.add_argument( "--image-size", dest="image_size", type=int, default=128, )
    parser.add_argument( "--add_size", type=int, help="How many patches should be added to the training set in each round", default=2, )
    parser.add_argument( "--offset", dest="offset", type=int, default=64, )
    parser.add_argument( "--seed", "-t", type=int, default=42, )
    parser.add_argument( "--validation", "-v", dest="val", type=int, default=25, help="Val percentage (0-100)", )
    parser.add_argument( "--export_results", type=int, default=0, help="If the added samples should be exported - this is for presentation slides", )
    parser.add_argument("--add_step", type=int, default=0, help = "> 0: examples will be added at preset intervals rather than considering the validation loss when validation set is sparsely annotated",)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for arg in vars(args):
        results[str(arg)] = getattr(args, arg)

    config = configparser.ConfigParser()
    # if config ini is in the current path, use it, otherwise look in parent folder
    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")
    save_path = config["PATHS"]["model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_net(device=device, args=args)
