import argparse
import configparser

import os


import platform
import my_data
from os.path import join as oj
import torch
import torch.nn as nn

# import wandb
from torch import optim
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

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
import wandb


is_windows = platform.system() == "Windows"
num_workers = 0 if is_windows else 4
wandb.init(project="VoxelSegmentWorking")

results = {}
np.random.seed()
file_name = "".join([str(np.random.choice(10)) for x in range(10)])
results["file_name"] = file_name
wandb.config["file_name"] = file_name


def train(
    net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, amp
):
    net.train()
    epoch_loss = 0

    for (
        images,
        true_masks,
    ) in train_loader:
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
    return epoch_loss / len(train_loader)


def train_net(device, args):

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
    init_train_idxs = all_train_idxs
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
    # num_train = len(train_set)
    x_pool, y_pool, _ = my_data.load_layer_data(
        oj(config["DATASET"]["data_path"], "lno")
    )

    x_pool, y_pool = x_pool[:-4], y_pool[:-4]

    all_idxs = np.arange(len(x_pool))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_train_pool = np.maximum(int(len(x) * (1 - args.val / 100)), 1)
    pool_ids = all_idxs[:n_train_pool]
    x_pool_fine, y_pool_fine = my_data.make_dataset(
        x_pool[pool_ids],
        y_pool[pool_ids],
        img_size=args.image_size,
        offset=args.image_size,
    )

    pool_set = TensorDataset(*[torch.Tensor(x_pool_fine), torch.Tensor(y_pool_fine),])

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
    initial_pool_len = len(pool_loader.dataset)

    old_num_train = len(train_loader.dataset)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    net = UNet(
        n_channels=1, n_classes=results["num_classes"], bilinear=args.bilinear
    ).to(device=device)
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
    delta = 0.0001
    patience = args.epochs  # no early stopping
    cur_patience = 0
    best_weights = None
    for epoch in range(1, args.epochs + 1):
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
        if val_score > best_val_score + delta or "our" in args.foldername:

            best_val_score = val_score
            best_weights = net.state_dict()
            cur_patience = 0
        else:
            cur_patience += 1
        if (
            epoch % args.add_step == 0 and len(pool_loader.dataset) / initial_pool_len > 1 - args.add_ratio
        ):
            old_num_train = len(train_loader.dataset)
            cur_patience = 0

            add_ids = cost_function(net, device, pool_loader, n_choose=8)
            add_set = TensorDataset(
                *[
                    torch.Tensor(x_pool_fine[add_ids]),
                    torch.Tensor(y_pool_fine[add_ids]),
                ]
            )
            newTrainSet = ConcatDataset([train_loader.dataset, add_set])
            train_loader = DataLoader(newTrainSet, shuffle=True, **loader_args)
            # delete from pool
            x_pool_fine = np.delete(x_pool_fine, add_ids, axis=0)
            y_pool_fine = np.delete(y_pool_fine, add_ids, axis=0)
            pool_set = TensorDataset(
                *[
                    torch.Tensor(x_pool_fine),
                    torch.Tensor(y_pool_fine),
                ]
            )
            pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)

            wandb.log(
                {
                    "dice_score": val_score,
                    "num_train": old_num_train,
                    "train_loss": train_loss,
                }
            )

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
        wandb.log(
            {
                "final_dice_score": results["final_dice_score"],
            }
        )

    pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

    torch.save(best_weights, oj(save_path, file_name + ".pt"))
    # wandb.alert(title="Run is done", text="Run is done")


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", type=int, default=1)
    parser.add_argument("--batch-size","-b", dest="batch_size", type=int, default=2,)
    parser.add_argument("--cost_function", dest="cost_function", type=str, default="random_cost",)
    parser.add_argument("--add_ratio", type=float, default=0.1,)
    parser.add_argument("--foldername", type=str, default="graphite_halfHour",)
    parser.add_argument("--experiment_name", "-g", type=str, default="",)
    parser.add_argument("--learningrate", "-l", type=float, default=1e-5, dest="lr",)
    parser.add_argument("--image-size", dest="image_size", type=int, default=128, )
    parser.add_argument("--offset", dest="offset", type=int, default=64,)
    parser.add_argument("--seed", "-t", type=int, default=42,)
    parser.add_argument("--scale", "-s", type=float, default=0.5, help="Downscaling factor of the images",)
    parser.add_argument("--validation", "-v", dest="val", type=int, default=10, help="Val percentage (0-100)", )
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--add_step", type=int, default=1)
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=6, help="Number of classes")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for arg in vars(args):

        results[str(arg)] = getattr(args, arg)
        wandb.config[str(arg)] = getattr(args, arg)
    config = configparser.ConfigParser()
    config.read("../config.ini")
    save_path = config["PATHS"]["model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_net(device= device, args=args)