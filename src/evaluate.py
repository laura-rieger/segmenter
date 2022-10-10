from json import load
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def random_cost_function(net, device, imgs, n_choose=1):
    idxs = np.arange(len(imgs))
    np.random.seed()
    np.random.shuffle(idxs)
    return idxs[-n_choose:]


def aq_cost_function_loader(net, device, loader, n_choose=-1):

    std_arr = -4 * np.ones((len(loader.dataset)))
    net.eval()
    with torch.no_grad():
        for i, (image, _) in tqdm(
            enumerate(loader),
            total=int(len(loader.dataset) / loader.batch_size),
            desc="AQ",
            unit="batch",
            leave=False,
        ):  # we only use the images, not the labels

            image = image.to(device)

            output = (
                F.softmax(net.forward(image), dim=1)
                .std(dim=1)
                .mean(axis=(1, 2))
                .detach()
                .cpu()
                .numpy()
            )

            std_arr[
                i * loader.batch_size : i * loader.batch_size + len(output)
            ] = output
            # std_arr[i] = output.max(axis=0).mean()  #mean
    # return std_arr
    return np.argsort(std_arr)[:n_choose]


def evaluate(net, dataloader, device, num_classes):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for (image, mask_true) in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        # image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # account for background

        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = (
                F.one_hot(mask_pred.argmax(dim=1), net.n_classes)
                .permute(0, 3, 1, 2)
                .float()
            )
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(
                mask_pred[:, :, ...],
                mask_true,
                num_classes,
                reduce_batch_first=False,
            )

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
