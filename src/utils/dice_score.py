import torch
from torch import Tensor
import numpy as np

def dice_coeff(
    input: Tensor, target: Tensor, target_is_mask, reduce_batch_first: bool = False, epsilon=1e-6, 
):
    # Average of Dice coefficient for all batches, or for a single mask
    # assert input.size() == target.size()

    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})"
        )

    if input.dim() == 2 or reduce_batch_first:
        input = input.reshape(-1)
        target = target.reshape(-1)
        target_is_mask = target_is_mask.reshape(-1)
        input = input[target_is_mask]
        target = target[target_is_mask]
        inter = torch.dot(input, target)
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...], target_is_mask[i])
        return dice / input.shape[0]



def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    num_classes: int,
    reduce_batch_first: bool = False,
    epsilon=1e-6, 
    separated_up = False
):
    # Average of Dice coefficient for all classes
    # assert input.size() == target.size()

    dice = 0
    for i,channel in enumerate(range(num_classes)):
        if not separated_up or i !=0:
            dice += dice_coeff(
                input[:, channel, ...],
                (target == channel).float(),
                target != 255,
                reduce_batch_first=reduce_batch_first,
                epsilon=epsilon,
            ) 
    if not separated_up:                                  
        return dice / input.shape[1]
    else:
        return dice / (input.shape[1]-1)


def dice_loss(input: Tensor, target: Tensor, num_classes, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # assert input.size() == target.size()

    return 1 - multiclass_dice_coeff(
        input, target, num_classes, reduce_batch_first=True
    )
