
import torch
import torch.nn.functional as F
# from tqdm import tqdm
import numpy as np
from utils.dice_score import multiclass_dice_coeff


def random_cost(net, device, loader, n_choose=-1):
    idxs = np.arange(len(loader.dataset))
    np.random.seed()
    np.random.shuffle(idxs)
    return idxs[-n_choose:]


def uncertainty_cost(net, device, loader, n_choose=-1):

    std_arr = -4 * np.ones((len(loader.dataset)))
    net.eval()
    with torch.no_grad():
        for i, image in enumerate(loader):  # we only use the images, not the labels

            image = image[0].to(device)
            output = F.softmax(net.forward(image), dim=1)
            entropy  = -torch.sum(output * torch.log(output), dim=1).mean(axis=(1,2)).detach().cpu().numpy()
 
            # output = (
            #     F.softmax(net.forward(image), dim=1)
            #     .std(dim=1)
            #     .mean(axis=(1, 2))
            #     .detach()
            #     .cpu()
            #     .numpy()
            # )

            std_arr[
                i * loader.batch_size: i * loader.batch_size + len(output)
            ] = entropy

    if n_choose == -1:
        return np.argsort(std_arr)
    else:
        return np.argsort(std_arr)[-n_choose:]
        # this doesn't make sense - 
        num_total = len(loader.dataset)
        offset_range = 0.05
        range = int(offset_range*num_total) 
        add_val = int((1-offset_range)*num_total)
        # randomly choose from the top 10% of the data
        np.random.seed(0)
        take_vals = np.random.choice(range, n_choose, replace=False)
        sorted_arr = np.argsort(std_arr)
        return np.take(sorted_arr, take_vals+add_val)

        # offset_val = 0



        # return np.argsort(std_arr)[-n_choose-offset_val:-offset_val]


#make a new function that takes in net, data loader, device and criterion and returns the loss
def evaluate_loss(net, device, loader, criterion,):
    tot_loss = 0
    net.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):  

            image = image.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)

            output = net.forward(image)
            loss = criterion(output, mask)
            tot_loss += loss.item()
    return tot_loss / len(loader.dataset)

def evaluate(net, dataloader, device, num_classes):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for (image, mask_true) in dataloader:
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
            # compute the Dice score, which is a metric for segmentation
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
