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
# import wandb
from torch import optim
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
# from tqdm import tqdm
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
# wandb.init(project="VoxelSegments")

results = {}
np.random.seed()
file_name = "".join([str(np.random.choice(10)) for x in range(10)])
results["file_name"] = file_name
# wandb.config["file_name"] = file_name


def train(
    net, train_loader, criterion, num_classes, optimizer, device, grad_scaler, amp
):
    net.train()
    epoch_loss = 0
    num_batches = 32

    for i,(
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

    return epoch_loss / (num_batches * train_loader.batch_size) # len(train_loader)


def train_net(device, args):
    loader_args = dict(batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)

    cost_function = getattr(evaluate, args.cost_function)
    results["val_scores"] = []
    # 1. Create dataset
    # val_scores = []
    x, y, num_classes, class_dict = my_data.load_layer_data(
        oj(config["DATASET"]["data_path"], args.foldername)
    )
    data_min, data_max = np.min(x[:-4]), np.max(x[:-4])
    x = (x - data_min) / (data_max - data_min)

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
    # if "lno" in args.foldername:
    #     # load the fully annotated data to get the final evaluation on unseen "real" data
    #     x_final, y_final, _, _ = my_data.load_layer_data(
    #         oj(config["DATASET"]["data_path"], "lno")
    #     )

    #     x_final, y_final = x_final[:-4], y_final[:-4]  # just don't touch the last four
    #     x_final = (x_final - data_min) / (data_max - data_min)
    #     all_idxs_final = np.arange(len(x_final))
    #     np.random.seed(0)
    #     np.random.shuffle(all_idxs_final)
    #     n_val_final = np.maximum(int(len(x_final) * args.val / 100), 1)
    #     val_idxs_final = all_idxs[-n_val_final:]
    #     val_set_final = TensorDataset(
    #         *[
    #             torch.Tensor(input)
    #             for input in my_data.make_dataset(
    #                 x_final[val_idxs_final],
    #                 y_final[val_idxs_final],
    #                 img_size=args.image_size,
    #                 offset=args.image_size,
    #             )
    #         ]
    #     )
    #     final_val_loader = DataLoader(val_set_final, shuffle=False, drop_last=True, **loader_args)
    # num_train = len(train_set)
    is_human_annotation = len(os.listdir(oj(config["DATASET"]["data_path"], args.poolname))) == 1

    if is_human_annotation:
        x_pool = my_data.load_pool_data(
            oj(config["DATASET"]["data_path"], args.poolname)
        )
        x_pool = x_pool[:-4]
        x_pool = (x_pool - data_min) / (data_max - data_min)
        all_idxs = np.arange(len(x_pool))
        np.random.seed(0)
        np.random.shuffle(all_idxs)
        n_train_pool = np.maximum(int(len(x) * (1 - args.val / 100)), 1)
        pool_ids = all_idxs[:n_train_pool]
        x_pool_all = my_data.make_dataset_single(
            x_pool[pool_ids],
            img_size=args.image_size,
            offset=args.image_size,
        )[0]

        pool_set = TensorDataset(torch.Tensor(x_pool_all))
    else:
        x_pool, y_pool, _, _ = my_data.load_layer_data(
            oj(config["DATASET"]["data_path"], args.poolname)
        )
        x_pool = (x_pool - data_min) / (data_max - data_min)

        x_pool, y_pool = x_pool[:-4], y_pool[:-4]

        all_idxs = np.arange(len(x_pool))
        np.random.seed(0)
        np.random.shuffle(all_idxs)
        n_train_pool = np.maximum(int(len(x) * (1 - args.val / 100)), 1)
        pool_ids = all_idxs[:n_train_pool]
        x_pool_all, y_pool_all = my_data.make_dataset(
            x_pool[pool_ids],
            y_pool[pool_ids],
            img_size=args.image_size,
            offset=args.image_size,
        )

        pool_set = TensorDataset(*[torch.Tensor(x_pool_all), torch.Tensor(y_pool_all), ])

    # 3. Create data loaders

    weights = [1 for x in range(len(train_set))]

    # Create a WeightedRandomSampler using the weights
    torch.manual_seed(args.seed)
    # sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    # train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
    initial_pool_len = len(pool_loader.dataset)

    # old_num_train = len(train_loader.dataset)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # val_loader = final_val_loader
    torch.manual_seed(args.seed)
    net = UNet(
        n_channels=1, n_classes=results["num_classes"], bilinear=args.bilinear
    ).to(device=device)
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    optimizer = optim.Adam(
        net.parameters(), lr=args.lr,
    )
    # optimizer = optim.SGD(
    #     net.parameters(), lr= args.lr, momentum=0.9,  nesterov=True
    # )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 5. Begin training
    best_val_score = 0
    # delta = 0.0001
    patience = 2
    cur_patience = 0
    best_weights = None
    epoch =0
    repetition_idx = 0
    while True:
    # for epoch in range(1, args.epochs + 1):
        _ = train(
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
        # val_score = evaluate.evaluate_loss(net, device, val_loader, criterion)
        results["val_scores"].append(val_score)
        print(val_score)
        sys.exit()
        # log the validation loss to wandb
        # wandb.log({"val_score": val_score})

        if val_score > best_val_score:
            print("New best validation score: " + str(val_score))

            best_val_score = val_score
            best_weights = deepcopy(net.state_dict())
            cur_patience = 0
        else:
            print(best_val_score)
            print(val_score)
            cur_patience += 1

        if cur_patience > patience or epoch == args.epochs:
            print("Ran out of patience, ")
            net.load_state_dict(best_weights)
            net.eval()


            if (len(pool_loader.dataset) > 0 and len(pool_loader.dataset) / initial_pool_len > 1 - args.add_ratio):
                cur_patience = 0
                add_ids = cost_function(net, device, pool_loader, n_choose=4)
                print("Added new to dataset:")

                if is_human_annotation:
                    my_data.save_progress(net,
                                          [add_ids, ],
                                          x_pool_all[add_ids],
                                          config["PATHS"]["progress_results"],
                                          file_name,
                                          args, device, results, class_dict)
                    print(file_name)
                    sys.exit()

                else:
                    add_set = TensorDataset(
                        *[
                            torch.Tensor(x_pool_all[add_ids]),
                            torch.Tensor(y_pool_all[add_ids]),
                        ]
                    )
                    # write out the new samples, the predictions and the labels for a presentation
                    if args.export_results == 1 and not is_human_annotation:
                        my_data.workflow_demo_save(net, x_pool_all[add_ids], y_pool_all[add_ids], config["PATHS"]["progress_results"], file_name, device, class_dict, repetition_idx)
                        repetition_idx += 1

                    newTrainSet = ConcatDataset([train_loader.dataset, add_set])
                    # ad hoc weigh the new samples ten times as much
                    new_weights = [1 for _ in range(len(train_loader.dataset))] + [5 for _ in range(len(add_set))]
                    new_sampler = torch.utils.data.WeightedRandomSampler(new_weights, len(new_weights))
                    train_loader = DataLoader(newTrainSet, sampler=new_sampler, **loader_args)
                    # delete from pool
                    x_pool_all = np.delete(x_pool_all, add_ids, axis=0)

                    y_pool_all = np.delete(y_pool_all, add_ids, axis=0)
                    pool_set = TensorDataset(*[torch.Tensor(x_pool_all),
                                              torch.Tensor(y_pool_all), ])

                    pool_loader = DataLoader(pool_set, shuffle=False, **loader_args)
                    print("Added {} samples to the training set".format(len(add_ids)))

            else:

                # load evaluation data
                if "lno" in args.foldername:

                    net.load_state_dict(best_weights)
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
                print(os.path.join(save_path, file_name + ".pkl"))
                pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

                torch.save(best_weights, oj(save_path, file_name + ".pt"))

                # wandb.alert(title="Run is done", text="Run is done")
                sys.exit()

        epoch += 1

def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=2,)
    parser.add_argument("--cost_function", dest="cost_function", type=str, default="uncertainty_cost",)
    parser.add_argument("--add_ratio", type=float, default=0.5,)
    parser.add_argument("--foldername", type=str, default="lno_halfHour",)

    parser.add_argument("--poolname", type=str, default="lno",)
    parser.add_argument("--experiment_name", "-g", type=str, default="",)
    parser.add_argument("--learningrate", "-l", type=float, default=0.001, dest="lr",)
    parser.add_argument("--image-size", dest="image_size", type=int, default=128, )
    parser.add_argument("--offset", dest="offset", type=int, default=64,)
    parser.add_argument("--seed", "-t", type=int, default=42,)
    parser.add_argument("--scale", "-s", type=float, default=0.5, help="Downscaling factor of the images",)
    parser.add_argument("--validation", "-v", dest="val", type=int, default=10, help="Val percentage (0-100)", )
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--export_results", type=int, default=1, help="If the added samples should be exported")
    # parser.add_argument("--add_step", type=int, default=1)
    # parser.add_argument("--human_annotation", action="store_true", default=True, help="Expect unannotated dataset")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=6, help="Number of classes")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for arg in vars(args):

        results[str(arg)] = getattr(args, arg)
        # wandb.config[str(arg)] = getattr(args, arg)
    config = configparser.ConfigParser()
    config.read("../config.ini")
    save_path = config["PATHS"]["model_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_net(device=device, args=args)