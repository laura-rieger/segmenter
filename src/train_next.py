import argparse
import configparser
import logging
import os
import sys
from pathlib import Path
import my_data
from os.path import join as oj
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch import optim
import pickle as pkl
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
#TODO Seed

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

results = {}
np.random.seed()
file_name = "".join([str(np.random.choice(10)) for x in range(10)])
results["file_name"] = file_name


def random_aqcuisition():
    pass


def active_aqcuisition():
    pass


def oracle_aqcuisition():
    pass


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
              init_train_ratio=1.0):
    # 1. Create dataset
    results['val_scores'] = []
    # val_scores = []
    in_imgs, in_targets = my_data.load_data(config['DATASET']['data_path'])
    in_imgs, in_targets = in_imgs[:
                                  -2], in_targets[:
                                                  -2]  #save the last two for test data
    x, y = my_data.make_rudimentary_dataset(
        (in_imgs, in_targets),
        img_size=image_size,
        offset=offset,
    )
    x = x[:, None]
    all_idxs = np.arange(len(x))
    np.random.seed(0)
    np.random.shuffle(all_idxs)
    n_val = int(len(x) * val_percent)
    n_train = len(x) - n_val

    all_train_idxs = all_idxs[:n_train]
    val_idxs = all_idxs[n_train:]
    init_train_idxs = all_train_idxs[:int(init_train_ratio * n_train)]

    pool_idxs = all_train_idxs[int(init_train_ratio * n_train):]

    train_set = TensorDataset(*[
        torch.Tensor(input)
        for input in [x[init_train_idxs], y[init_train_idxs]]
    ])
    val_set = TensorDataset(
        *[torch.Tensor(input) for input in [x[val_idxs], y[val_idxs]]])
    num_train = len(train_set)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    best_val_score = 0
    delta = .001
    best_weights = None
    patience = 3
    cur_patience = 0
    division_step = np.maximum((num_train // (10 * batch_size)), 10)

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=num_train, desc=f'Epoch {epoch}/{epochs}',
                  unit='img') as pbar:
            for (
                    images,
                    true_masks,
            ) in train_loader:

                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

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
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round

                if global_step % division_step == 0:

                    val_score = evaluate(net, val_loader, device).item()
                    results['val_scores'].append(val_score)
                    scheduler.step(val_score)
                    if val_score > best_val_score + delta:
                        best_val_score = val_score
                        best_weights = net.state_dict()
                        cur_patience = 0
                    else:
                        cur_patience += 1

                    logging.info('Validation Dice score: {}'.format(val_score))
                if cur_patience > patience:
                    break

    pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

    torch.save(best_weights, oj(save_path, file_name + ".pt"))
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    logging.info(f'Checkpoint {epoch} saved!')


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
                        default=32,
                        help='Batch size')

    parser.add_argument(
        '--init_train_ratio',
        dest='init_train_ratio',
        type=float,
        default=1.0,
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
    parser.add_argument('--load',
                        '-f',
                        type=str,
                        default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--scale',
                        '-s',
                        type=float,
                        default=1.0,
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
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(
        f'Network:\n'
        f'\t{net.n_channels} input channels\n'
        f'\t{net.n_classes} output channels (classes)\n'
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
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
                  init_train_ratio=args.init_train_ratio)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise