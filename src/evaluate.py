
import torch
import torch.nn.functional as F
# from tqdm import tqdm
import my_data
import numpy as np
from utils.dice_score import multiclass_dice_coeff
def final_evaluate(net, x_test, y_test, num_classes, device):

    net.eval()
 


    y_pred = []
    for i in range(len(x_test)):
        y_pred_nu = net(torch.Tensor(x_test[i]).unsqueeze(0).to(device)).cpu().detach().numpy()
        y_pred.append(y_pred_nu)
    y_pred = np.asarray(y_pred).squeeze().argmax(axis=1)
    y_pred_one_hot = torch.nn.functional.one_hot(torch.Tensor(y_pred).to(torch.int64), 
                                                num_classes=num_classes).permute(0, 3, 1, 2)
    result = multiclass_dice_coeff(y_pred_one_hot.float(), 
                        torch.Tensor(y_test), 
                        num_classes=num_classes).item()
        
    return result

def random_cost(net, device, loader, n_choose=-1):
    idxs = np.arange(len(loader.dataset))
    np.random.seed()
    np.random.shuffle(idxs)
    return idxs[-n_choose:]


def num_pixel_cost(net, device, loader, percentile=.5,  n_choose=-1):

    std_arr = -4 * np.ones((len(loader.dataset)))
    net.eval()
    with torch.no_grad():
        for i, image in enumerate(loader):  # we only use the images, not the labels

            image = image[0].to(device)
            output = F.softmax(net.forward(image), dim=1)[:,:, 2:-2, 2:-2]
            entropy  = -torch.sum(output * torch.log(output), dim=1)
            # compute the 50 percentile for each image in torch
            entropy_reshaped = entropy.reshape(entropy.shape[0], -1)
            # sort the values
            
            std_arr[
                i * loader.batch_size: i * loader.batch_size + len(output)
            ] = (entropy_reshaped > .1).float().mean(dim=1).cpu().numpy()

    if n_choose == -1:
        return np.argsort(std_arr)
    else:
        return np.argsort(std_arr)[-n_choose:]


def cut_off_cost(net, device, loader, percentile=.5,  n_choose=-1):

    std_arr = -4 * np.ones((len(loader.dataset)))
    net.eval()
    with torch.no_grad():
        for i, image in enumerate(loader):  # we only use the images, not the labels

            image = image[0].to(device)
            output = F.softmax(net.forward(image), dim=1)[:,:, 2:-2, 2:-2]
            entropy  = -torch.sum(output * torch.log(output), dim=1)
            # compute the 50 percentile for each image in torch
            entropy_reshaped = entropy.reshape(entropy.shape[0], -1)
            # sort the values
            entropy_sorted = torch.sort(entropy_reshaped, dim=1)[0]
            # get the 50th percentile
            num_used = int(entropy_sorted.shape[1]*percentile)
            std_arr[
                i * loader.batch_size: i * loader.batch_size + len(output)
            ] = entropy_reshaped[:, -num_used:].mean(axis=(1)).detach().cpu().numpy()

    if n_choose == -1:
        return np.argsort(std_arr)
    else:
        #upper five percent
      
        up_five = int(len(loader.dataset) * .05)
        pot_idxs = np.argsort(std_arr)[-up_five:]
        #randomly choose n_choose from the upper five percent
        # np.random.seed()
        np.random.shuffle(pot_idxs)
        return pot_idxs[:n_choose]
        return np.argsort(std_arr)[-n_choose:]
# def give_results(net, device, loader, ):
#     output_list = []
#     net.eval()
#     with torch.no_grad():
#         for i, image in enumerate(loader):  # we only use the images, not the labels

#             image = image[0].to(device)
#             output = F.softmax(net.forward(image), dim=1).argmax(dim= 1)
#             # print(output.shape)
#             output_list.append(output.detach().cpu().numpy())


#     return np.asarray(output_list).squeeze()




#make a new function that takes in net, data loader, device and criterion and returns the loss
# def evaluate_loss(net, device, loader, criterion):
#     tot_loss = 0
#     net.eval()
#     with torch.no_grad():
#         for i, (image, mask) in enumerate(loader):  

#             image = image.to(device, dtype=torch.float32)
#             mask = mask.to(device, dtype=torch.long)

#             output = net.forward(image)
#             loss = criterion(output, mask)
#             tot_loss += loss.item()
#     return tot_loss / len(loader.dataset)

def evaluate(net, dataloader, device, num_classes, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set

    for (image, mask_true) in dataloader:

        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():

            mask_pred = net(image)
            mask_pred_out = (
                F.one_hot(mask_pred.argmax(dim=1), net.n_classes)
                .permute(0, 3, 1, 2)
                .float()
            )
        
            # compute the Dice score, which is a metric for segmentation
            dice_score += multiclass_dice_coeff(
                mask_pred_out[:, :, ...],
                mask_true,
                num_classes,
                reduce_batch_first=False,
            ).item()

    net.train()

    if num_val_batches == 0:
        return (dice_score)
    # print(crit_loss* dataloader.batch_size/num_val_batches, dice_score/ num_val_batches)
    
    return dice_score / num_val_batches
