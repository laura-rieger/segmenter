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


def aq_cost_function(net, device, imgs, n_choose=1):
    logsoftmax = torch.nn.LogSoftmax(dim=0)
    std_arr = np.zeros((len(imgs)))
    net.eval()
    with torch.no_grad():
        for i in range(len(imgs)):

            img_t = torch.Tensor(imgs[i][None, None, :]).to(device)

            output = net.forward(img_t)  #.cpu().detach().numpy()[0]
            # return output
            # print(output.shape)
            # std = np.quantile(output.std(axis=0), .1)  #Std

            std_arr[i] = (logsoftmax(output[0]) *
                          torch.softmax(output[0], axis=0)).mean().item()
            # std_arr[i] = output.max(axis=0).mean()  #mean
    return np.argsort(std_arr)[-n_choose:]


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for (image, mask_true) in tqdm(dataloader,
                                   total=num_val_batches,
                                   desc='Validation round',
                                   unit='batch',
                                   leave=False):
        # image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1,
                                                                2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true,
                                         reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1),
                                      net.n_classes).permute(0, 3, 1,
                                                             2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...],
                                                    mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
