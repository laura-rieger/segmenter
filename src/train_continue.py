import argparse
from asyncio import run
import configparser

import os


import platform
import my_data
from os.path import join as oj
import torch
import torch.nn as nn
import sys

from torch import optim
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F

# from torch.utils.data import DataLoader, TensorDataset

from utils.dice_score import dice_loss
import evaluate
# from evaluate import (
#     evaluate,
#     random_cost_function,
#     std_cost_function,
# )
from unet import UNet
# import wandb


is_windows = platform.system() == "Windows"
num_workers = 0 if is_windows else 4
# wandb.init(project="VoxelSegmentWorking")

results = {}

file_name = -1
# results["file_name"] = file_name

def train(
    net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, amp
):
    num_batches = 64
    net.train()
    epoch_loss = 0

    for i, (
        images,
        true_masks,
    ) in enumerate(train_loader):
        if np.random.uniform() > 0.5:
            images = torch.flip(images, [2, ],)
            true_masks = torch.flip(true_masks, [1, ],)
        if np.random.uniform() > 0.5:
            images = torch.flip(images, [3, ],)
            true_masks = torch.flip(true_masks, [2, ],)
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=amp):

            if torch.any(true_masks != 255):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks)

                loss_dice = dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    true_masks,
                    num_classes,
                    multiclass=True,
                )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss + loss_dice).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()
        if i >= num_batches:
            break
    return epoch_loss / (num_batches*train_loader.batch_size)


def train_net(device, args, run_id):
    cost_function = getattr(evaluate, args.cost_function)
    results["val_scores"] = []
    # 1. Create dataset
    # val_scores = []
    x, y, num_classes = my_data.load_layer_data(
        oj(config["DATASET"]["data_path"], args.foldername)
    )

    results["num_classes"] = num_classes
    x, y = x[:-4], y[:-4]  # just don't touch the last four

    all_idxs = np.arange(len(x))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_val = np.maximum(int(len(x) * args.val / 100), 1)
    n_train = len(x) - n_val
    all_train_idxs = all_idxs[:n_train]
    val_idxs = all_idxs[n_train:]
    init_train_idxs = all_train_idxs[:1] # xxx remove
    # pool_idxs = all_train_idxs[np.maximum(1, int(init_train_ratio * n_train)) :]

    train_set = TensorDataset(
        *[
            torch.Tensor(input)
            for input in my_data.make_dataset(
                x[init_train_idxs],
                y[init_train_idxs],
                img_size=args.image_size,
                offset=args.offset,
            )
        ]
    )
    # old_num_train = len(train_set)
    annotated_set = my_data.load_annotated_imgs(oj(config["PATHS"]["progress_results"], run_id))
    weights = [1 for _ in range(len(train_set))] + [10 for _ in range(len(annotated_set))]

    train_set = ConcatDataset([train_set, annotated_set])
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
    # is_human_annotation = len(os.listdir(oj(config["DATASET"]["data_path"], "lno"))) ==1

    x_pool = my_data.load_pool_data(
        oj(config["DATASET"]["data_path"], args.poolname)
    )
    x_pool = x_pool[:-4]
    all_idxs = np.arange(len(x_pool))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_train_pool = np.maximum(int(len(x) * (1 - args.val / 100)), 1)
    pool_ids_init = all_idxs[:n_train_pool]
    
    x_pool_fine = my_data.make_dataset_single(
        x_pool[pool_ids_init],
        img_size=args.image_size,
        offset=args.image_size,
    )[0]
    pool_ids = np.arange(len(x_pool_fine))
    remove_id_list = pkl.load(open(oj(config["PATHS"]["progress_results"], run_id, 'image_idxs.pkl'), "rb"))
    for remove_ids in remove_id_list:
        pool_ids = np.delete(pool_ids, remove_ids, axis=0)
    
    x_pool_fine = x_pool_fine[pool_ids]
    pool_set = TensorDataset(*[torch.Tensor(x_pool_fine), ])

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)

    


    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_set, sampler = sampler, **loader_args)
    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
    initial_pool_len = len(pool_set) + len(annotated_set)

    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    net = UNet(
        n_channels=1, n_classes=results["num_classes"], bilinear=args.bilinear
    ).to(device=device)
    net.load_state_dict(torch.load(oj(config["PATHS"]["progress_results"], run_id, "model_state.pt")))
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        net.parameters(), lr=args.lr, weight_decay=0, momentum=0.0
    )
    optimizer = optim.Adamax(
        net.parameters(),
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 5. Begin training
    best_val_score = 0
    patience = args.epochs  # no early stopping
    cur_patience = 0
    best_weights = None
    init_epochs = len(results['val_scores'])
    for epoch in tqdm(range(init_epochs, args.epochs + 1)):
        train_loss = train(
            net,
            train_loader,
            criterion,
            num_classes,
            optimizer,
            device,
            grad_scaler,
            args.amp,
        )
    
        val_score = evaluate.evaluate(net, val_loader, device, num_classes).item()
        results["val_scores"].append(val_score)

        # if the dataset is the roughly annotated one, we just train until the end (those are annotated with hour/Hour)
        if val_score > best_val_score or "our" in args.foldername:

            best_val_score = val_score
            best_weights = net.state_dict()
            cur_patience = 0
        else:
            cur_patience += 1
        if (
            epoch % args.add_step == 0
            and len(pool_loader.dataset) / initial_pool_len > 1 - args.add_ratio
        ):
      
            add_ids = cost_function(net, device, pool_loader, n_choose=8)
            remove_id_list.append(add_ids)
            my_data.save_progress(net,
                                  remove_id_list, 
                                  x_pool_fine[add_ids], 
                                  config["PATHS"]["progress_results"],
                                  file_name, args, device, results)
            print(file_name)
            sys.exit()

        if cur_patience > patience:

            break

    # do a final evaluation
    net.load_state_dict(best_weights)
    # load evaluation data
    if "lno" in args.foldername:
        x, y, num_classes = my_data.load_layer_data(
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
        # wandb.log(
        #     {
        #         "final_dice_score": results["final_dice_score"],
        #     }
        # )

    pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

    torch.save(best_weights, oj(save_path, file_name + ".pt"))
    # wandb.alert(title="Run is done", text="Run is done")


def get_args():
    parser = argparse.ArgumentParser(description="Train a unet ")
    parser.add_argument("--filename", "-e", type=str, default='9622280261')
    return parser.parse_args()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("../config.ini")
    args = get_args()
    # load args
    run_id = args.filename
    file_name = run_id
    results["file_name"] = file_name
    # wandb.config["file_name"] = file_name

    run_folder = oj(config["PATHS"]["progress_results"], str(run_id))
    args = pkl.load(open(oj(run_folder, "args.pkl"), "rb"))
    results = pkl.load(open(oj(run_folder, "results.pkl"), "rb"))
    for arg in vars(args):
        results[str(arg)] = getattr(args, arg)
        # wandb.config[str(arg)] = getattr(args, arg)
    save_path = config["PATHS"]["model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_net(device=device, args=args,run_id =run_id)