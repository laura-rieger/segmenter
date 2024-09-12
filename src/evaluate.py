import torch
import torch.nn.functional as F
# from tqdm import tqdm
import my_data
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from predict import run
from utils.dice_score import multiclass_dice_coeff

def final_evaluate(net, x_test, y_test, num_classes, device, separated_up = False):

    net.eval()
 

    y_pred = run(net, x_test, 0, 1, 256, num_classes,use_orig_values = True   ) # xxx

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[None, :]

    y_pred_one_hot = torch.nn.functional.one_hot(torch.Tensor(y_pred).to(torch.int64), 
                                                num_classes=num_classes).permute(0, 3, 1, 2)

    y_test = torch.Tensor(y_test).to(torch.int64)
    if len(y_test.shape) == 2:
        y_test = y_test[None, :]

    result = multiclass_dice_coeff(y_pred_one_hot.float(), 
                        torch.Tensor(y_test), 
                        num_classes=num_classes, separated_up= separated_up).item()
    # denom = (y_pred_one_hot.shape[1]-1) if separated_up else y_pred_one_hot.shape[1] # account for some classes not being there and having 0 in dice score
    # nom =len(np.unique(y_test)) if separated_up else len(np.unique(y_test)) -1
    # result = result*denom / nom
        
    return result

def random_cost(net, device, loader,data_vals, n_choose=-1, num_classes = None,criterion = None,):
    idxs = np.arange(len(loader.dataset))
    np.random.seed()
    np.random.shuffle(idxs)
    return idxs[-n_choose:]


def emc(net, device, loader, data_vals, n_choose=-1, num_classes = None, criterion = None, ):
    new_loader = DataLoader(loader.dataset, batch_size=1, shuffle=False, )
    (data_min, data_max) = data_vals
    grad_scaler = torch.cuda.amp.GradScaler()
    std_arr = -4 * np.ones((len(loader.dataset)))


    for i, image in enumerate(new_loader):  # we only use the images, not the labels



        image_in = ((image[0].float() - data_min)/(data_max-data_min)).to(device)
        tot_loss = 0
        for j in range(num_classes):
            # net.zero_grad(set_to_none=True) 
            net.zero_grad()
            masks_pred = F.softmax(net.forward(image_in), dim=1)
            masks_pred = masks_pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
            true_masks = j * torch.ones(len(masks_pred), dtype = torch.int64).to(device=device, dtype=torch.long)

            loss = criterion(masks_pred, true_masks)
            grad_scaler.scale(loss ).backward()
 
  
            for param in net.parameters():
                if param.grad is not None:
                    tot_loss += torch.abs(param.grad).sum()
              

        std_arr[
                i * new_loader.batch_size: i * new_loader.batch_size + len(masks_pred)
            ] = tot_loss.cpu().detach().numpy()

    if n_choose == -1:
        return np.argsort(std_arr)
    else:
        return np.argsort(std_arr)[-n_choose:]
 
        
def cut_off_cost(net, device, loader, data_vals, percentile=.5,  n_choose=-1,num_classes = None,criterion = None,):
    print(percentile)
    (data_min, data_max) = data_vals
    std_arr = -4 * np.ones((len(loader.dataset)))
    net.eval()
    
    with torch.no_grad():
        for i, image in enumerate(loader):  # we only use the images, not the labels

            image = ((image[0].float() - data_min)/(data_max-data_min)).to(device)
            output = F.softmax(net.forward(image), dim=1)[:,:, 2:-2, 2:-2]
            entropy  = -torch.sum(output * torch.log(output), dim=1)
            # compute the 50 percentile for each image in torch
            entropy_reshaped = entropy.reshape(entropy.shape[0], -1)
     
            entropy_sorted = torch.sort(entropy_reshaped, dim=1)[0]
            # get the 50th percentile
            num_used = int(entropy_sorted.shape[1]*percentile)
            std_arr[
                i * loader.batch_size: i * loader.batch_size + len(output)
            ] = entropy_reshaped[:, -num_used:].mean(axis=(1)).detach().cpu().numpy()

    if n_choose == -1:
        return np.argsort(std_arr)
    else:
        up_five = int(len(loader.dataset) * .05)
        pot_idxs = np.argsort(std_arr)[-up_five:]
        np.random.seed()
        np.random.shuffle(pot_idxs)
        return pot_idxs[:n_choose]



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
            mask_pred_out = ( F.one_hot(mask_pred.argmax(dim=1), num_classes) .permute(0, 3, 1, 2) .float() )
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
