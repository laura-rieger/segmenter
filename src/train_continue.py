import argparse
import configparser
import os
import platform
import my_data
from os.path import join as oj
import torch
import torch.nn as nn
import sys
from copy import deepcopy
from torch import optim
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from utils.dice_score import dice_loss
import evaluate
from unet import UNet

is_windows = platform.system() == "Windows"
num_workers = 0 if is_windows else 4

# results = {}

file_name = -1

def train(net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, ):
    num_batches = 64
    net.train()
    epoch_loss = 0

    for i, (images, true_masks, ) in enumerate(train_loader):
        if np.random.uniform() > 0.5:
            images = torch.flip(images, [2, ],)
            true_masks = torch.flip(true_masks, [1, ],)
        if np.random.uniform() > 0.5:
            images = torch.flip(images, [3, ],)
            true_masks = torch.flip(true_masks, [2, ],)
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=True):

            if torch.any(true_masks != 255):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks)
                loss_dice = dice_loss(F.softmax(masks_pred, dim=1).float(), true_masks, num_classes, multiclass=True, )
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss + loss_dice).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()
        if i >= num_batches:
            break
    return epoch_loss / (num_batches*train_loader.batch_size)


def train_net(device, args, run_id, results ):
    cost_function = getattr(evaluate, args.cost_function)
    # results["val_scores"] = []
    # 1. Create dataset
    x, y, num_classes, class_dict = my_data.load_layer_data( oj(config["DATASET"]["data_path"], args.foldername))
    data_min, data_max = np.min(x[:-4]), np.max(x[:-4])
    # results["num_classes"] = num_classes
    x, y = x[:-4], y[:-4]  #  don't touch the last full image - left for test
    
    x = (x - data_min) / (data_max - data_min)

    all_idxs = np.arange(len(x))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_val = np.maximum(int(len(x) * args.val / 100), 1)
    n_train = len(x) - n_val
    all_train_idxs = all_idxs[:n_train]
    val_idxs = all_idxs[n_train:]
    init_train_idxs = all_train_idxs 
    train_set = TensorDataset( *[ torch.Tensor(input) for input in my_data.make_dataset( x[init_train_idxs], y[init_train_idxs], img_size=args.image_size, offset=args.offset, ) ] )

    annotated_set = my_data.load_annotated_imgs(oj(config["PATHS"]["progress_results"], run_id, ), class_dict)
    weights = [1 for _ in range(len(train_set))] + [10 for _ in range(len(annotated_set))]

    train_set = ConcatDataset([train_set, annotated_set])
    val_set = TensorDataset( *[ torch.Tensor(input) for input in my_data.make_dataset( x[val_idxs], y[val_idxs], img_size=args.image_size, offset=args.image_size, ) ] )
    x_pool = my_data.load_pool_data(
        oj(config["DATASET"]["data_path"], args.poolname)
    )
    x_pool = x_pool.astype(np.float16)
    x_pool = (x_pool - data_min) / (data_max - data_min)
    x_pool_fine = my_data.make_dataset_single( x_pool, img_size=args.image_size, offset=args.image_size, )[0]
    pool_ids = np.arange(len(x_pool_fine))
    remove_id_list = pkl.load(open(oj(config["PATHS"]["progress_results"], run_id, 'image_idxs.pkl'), "rb"))
    for remove_ids in remove_id_list:
        pool_ids = np.delete(pool_ids, remove_ids, axis=0)
    x_pool_dataset = x_pool_dataset[pool_ids]
    pool_set = TensorDataset(*[torch.Tensor(x_pool_dataset), ])
    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
    initial_pool_len = len(pool_set) + len(annotated_set)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    net = UNet(n_channels=1, n_classes=results["num_classes"],).to(device=device)
    net.load_state_dict(torch.load(oj(config["PATHS"]["progress_results"], run_id, "model_state.pt")))
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam( net.parameters(),args.lr, )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # 5. Begin training
    best_val_score = 0
    patience = 3  
    cur_patience = 0
    best_weights = None
    init_epochs = len(results['val_scores'])
    # args.epochs = 10000
    for epoch in tqdm(range(init_epochs, args.epochs + 1)):
        train_loss =   train(net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, )
        val_score = evaluate.evaluate(net, val_loader, device, num_classes).item()
        results["val_scores"].append(val_score)
        args.epochs = 10000
        # results["train_losses"].append(train_loss)

        if val_score > best_val_score and args.add_step == 0:

            best_val_score = val_score
            best_weights = deepcopy(net.state_dict())
            cur_patience = 0
        else:
            cur_patience += 1

        # if the add step is not zero, just go after add step
        default_add = args.add_step != 0 and epoch % args.add_step == 0 
        patience_add = args.add_step == 0 and cur_patience >= patience 
        label_budget_exceeded = len(pool_loader.dataset) / initial_pool_len < 1 - args.add_ratio
        if (default_add or patience_add) and not label_budget_exceeded:
        
            add_ids = cost_function(net, device, pool_loader, n_choose=args.add_size)
            remove_id_list.append(add_ids)
            my_data.save_progress(net, remove_id_list, x_pool_dataset[add_ids], config["PATHS"]["progress_results"], file_name, args, device, results, class_dict)
            print(file_name)
            sys.exit()

      

        if cur_patience >= patience and label_budget_exceeded:
            break

    # do a final evaluation
    if best_weights is not None:
        net.load_state_dict(best_weights)
    net.eval()
    # load evaluation data
    if os.path.exists(oj(config["DATASET"]["data_path"], "lno")) and ("lno" in args.foldername or "LNO" in args.foldername):
        x, y, num_classes, _ = my_data.load_layer_data(
            oj(config["DATASET"]["data_path"], "lno")
        )
        x, y = x[:-4], y[:-4]  # just don't touch the last four
        all_idxs = np.arange(len(x))
        np.random.seed(0)
        np.random.shuffle(all_idxs)
        n_val = np.maximum(int(len(x) * args.val / 100), 1)
        val_idxs = all_idxs[-n_val:]
        val_set = TensorDataset(
            *[
                torch.Tensor(input)
                for input in my_data.make_dataset(
                    x[val_idxs],
                    y[val_idxs],
                    img_size=args.image_size,
                    offset=args.image_size,
                )
            ]
        )
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        final_dice_score = evaluate.evaluate(net, val_loader, device, num_classes)
        if type(final_dice_score) == torch.Tensor:
            results["final_dice_score"] = final_dice_score.item()
        else:
            results["final_dice_score"] = final_dice_score
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

    torch.save(net.state_dict(), oj(save_path, file_name + ".pt"))



def get_args():
    parser = argparse.ArgumentParser(description="Train a unet ")
    parser.add_argument("--filename", "-e", type=str, default='9772197289')
    return parser.parse_args()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    # if there is a config file in the parent folder load it, otherwise look in this folder
    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")
    args = get_args()
    # load args
    run_id = args.filename
    file_name = run_id
    


    run_folder = oj(config["PATHS"]["progress_results"], str(run_id))
    args = pkl.load(open(oj(run_folder, "args.pkl"), "rb"))
    if is_windows:
        args.batch_size = 2
    
    results = pkl.load(open(oj(run_folder, "results.pkl"), "rb"))
    for arg in vars(args):
        results[str(arg)] = getattr(args, arg)
    # results["file_name"] = file_name
    save_path = config["PATHS"]["model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_net(device=device, args=args, run_id =run_id, results =  results )