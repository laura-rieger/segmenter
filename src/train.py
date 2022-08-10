import argparse
import configparser
import logging
import os
from re import A
import sys
import torchvision.transforms.functional as TF
from pathlib import Path
import platform
import my_data
from os.path import join as oj
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch import optim
import pickle as pkl
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
from tqdm import tqdm
import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
from utils.dice_score import dice_loss
from evaluate import evaluate, aq_cost_function, random_cost_function
from unet import UNet
import wandb

is_windows = platform.system() == 'Windows'
num_workers = 0 if is_windows else 4
wandb.init(project="VoxelSegment")

results = {}
np.random.seed()
file_name = "".join([str(np.random.choice(10)) for x in range(10)])
results["file_name"] = file_name
wandb.config['file_name'] = file_name


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.25,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              image_size=128,
              offset=64,
              amp: bool = False,
              init_train_ratio=1.0,
              final_ratio=.75,
              add_step=0,
              cost_funct=random_cost_function):
    # 1. Create dataset
    results['val_scores'] = []
    # val_scores = []
    x, y = my_data.load_layer_data(config['DATASET']['data_layer_path'])
    # data = data[:-4]  # just don't touch the last four

    all_idxs = np.arange(len(x))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_val = np.maximum(int(len(x) * val_percent), 1)
    n_train = len(x) - n_val
    all_train_idxs = all_idxs[:n_train]
    val_idxs = all_idxs[n_train:]
    init_train_idxs = all_train_idxs[:np.
                                     maximum(1, int(init_train_ratio *
                                                    n_train))]
    pool_idxs = all_train_idxs[np.maximum(1, int(init_train_ratio * n_train)):]

    train_set = TensorDataset(*[
        torch.Tensor(input) for input in my_data.make_dataset(
            x[init_train_idxs],
            y[init_train_idxs],
            img_size=image_size,
            offset=offset,
        )
    ])
    val_set = TensorDataset(*[
        torch.Tensor(input) for input in my_data.make_dataset(
            x[val_idxs],
            y[val_idxs],
            img_size=image_size,
            offset=offset,
        )
    ])
    num_train = len(train_set)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size,
                       num_workers=num_workers,
                       pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    old_num_train = len(train_loader.dataset)
    val_loader = DataLoader(val_set,
                            shuffle=False,
                            drop_last=True,
                            **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {num_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(),
                              lr=learning_rate,
                              weight_decay=1e-8,
                              momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    best_val_score = 0
    delta = .001
    patience = 5
    cur_patience = 0
    best_weights = None

    break_cond = False

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=num_train, desc=f'Epoch {epoch}/{epochs}',
                  unit='img') as pbar:
            for (
                    images,
                    true_masks,
            ) in train_loader:
                if np.random.uniform() > 5:
                    images = torch.flip(images, 2)
                    true_masks = torch.flip(true_masks, 1)
                if np.random.uniform() > 5:
                    images = torch.flip(images, 3)
                    true_masks = torch.flip(true_masks, 2)

                images = images.to(device=device, dtype=torch.float32)

                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) + dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, net.n_classes).permute(
                            0, 3, 1, 2).float(),
                        multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if global_step % add_step == 0:

                    val_score = evaluate(net, val_loader, device).item()
                    results['val_scores'].append(val_score)
                    # scheduler.step(val_score)
                    if val_score > best_val_score + delta:
                        best_val_score = val_score
                        best_weights = net.state_dict()
                        cur_patience = 0
                    else:
                        cur_patience += 1
                    if add_step > 0 and len(pool_idxs) > 0 and len(
                            pool_idxs) >= (len(init_train_idxs) +
                                           len(pool_idxs) * final_ratio):
                        old_num_train = len(train_loader.dataset)
                        cur_patience = 0

                        add_ids = cost_funct(net, device, data[pool_idxs, 0])
                        add_set = TensorDataset(*[
                            torch.Tensor(input)
                            for input in my_data.make_dataset(
                                data[pool_idxs[add_ids]],
                                img_size=image_size,
                                offset=offset,
                            )
                        ])
                        newTrainSet = ConcatDataset(
                            [train_loader.dataset, add_set])
                        train_loader = DataLoader(newTrainSet,
                                                  shuffle=True,
                                                  **loader_args)
                        init_train_idxs = np.concatenate(
                            [init_train_idxs, pool_idxs[add_ids]])

                        pool_idxs = np.delete(pool_idxs,
                                              add_ids)  #remove from pool
                    wandb.log({
                        'dice_score':
                        val_score,
                        'num_train':
                        old_num_train,
                        'train_loss':
                        epoch_loss / (add_step * batch_size)
                    })

                    epoch_loss = 0

                if cur_patience > patience:
                    break_cond = True
                    break

            if break_cond:
                break
        if break_cond:
            break

    # pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

    # torch.save(best_weights, oj(save_path, file_name + ".pt"))
    # wandb.alert(title="Run is done", text="Run is done")


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--epochs',
                        '-e',
                        metavar='E',
                        type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size',
                        '-b',
                        dest='batch_size',
                        metavar='B',
                        type=int,
                        default=2,
                        help='Batch size')
    parser.add_argument(
        '--cost_function',
        dest='cost_function',
        type=str,
        default="Test",
    )
    parser.add_argument(
        '--init_train_ratio',
        dest='init_train_ratio',
        type=float,
        default=.3,
    )
    parser.add_argument('--experiment_name',
                        '-g',
                        dest='experiment_name',
                        metavar='G',
                        type=str,
                        default="",
                        help='Name')
    parser.add_argument('--learning-rate',
                        '-l',
                        metavar='LR',
                        type=float,
                        default=1e-5,
                        help='Learning rate',
                        dest='lr')
    parser.add_argument('--image-size',
                        dest='image_size',
                        type=int,
                        default=128,
                        help='Image size')
    parser.add_argument('--offset',
                        dest='offset',
                        type=int,
                        default=64,
                        help='Offset')
    parser.add_argument('--seed', '-t', type=int, default=42, help='Seed')

    parser.add_argument('--scale',
                        '-s',
                        type=float,
                        default=.5,
                        help='Downscaling factor of the images')
    parser.add_argument(
        '--validation',
        '-v',
        dest='val',
        type=float,
        default=10.0,
        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help='Use mixed precision')
    parser.add_argument('--add_step', type=int, default=1)
    parser.add_argument('--bilinear',
                        action='store_true',
                        default=False,
                        help='Use bilinear upsampling')
    parser.add_argument('--classes',
                        '-c',
                        type=int,
                        default=3,
                        help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    for arg in vars(args):
        if arg != "save_path":
            results[str(arg)] = getattr(args, arg)
            wandb.config[str(arg)] = getattr(args, arg)

    config = configparser.ConfigParser()
    config.read("../config.ini")

    save_path = config["PATHS"]["model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net = UNet(n_channels=5, n_classes=args.classes, bilinear=args.bilinear)
    if args.cost_function == "Random":
        cost_function = random_cost_function
    else:

        cost_function = aq_cost_function

    logging.info(
        f'Network:\n'
        f'\t{net.n_channels} input channels\n'
        f'\t{net.n_classes} output channels (classes)\n'
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.to(device=device)

    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batch_size,
              learning_rate=args.lr,
              image_size=args.image_size,
              offset=args.offset,
              device=device,
              img_scale=args.scale,
              val_percent=args.val / 100,
              amp=args.amp,
              add_step=args.add_step,
              init_train_ratio=args.init_train_ratio,
              cost_funct=cost_function)
