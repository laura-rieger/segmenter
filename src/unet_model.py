""" Full assembly of the parts to form the complete network """
# from https://github.com/milesial/Pytorch-UNet 
import torch.nn.functional as F
import numpy as np
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, in_factor =32, use_small = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.use_small = use_small

        self.inc = DoubleConv(n_channels, in_factor)
        self.down1 = Down(in_factor, in_factor*2)
        self.down2 = Down(in_factor*2, in_factor*4)
        
        if not self.use_small:
            self.down3 = Down(in_factor*4, in_factor*8)
            self.down4 = Down(in_factor*8, in_factor*16 // factor)
            self.up1 = Up(in_factor*16, in_factor*8 // factor, bilinear)
        else:
            self.down3 = Down(in_factor*4, in_factor*8//factor)
        self.up2 = Up(in_factor*8, in_factor*4 // factor, bilinear)
        self.up3 = Up(in_factor*4, in_factor*2 // factor, bilinear)
        self.up4 = Up(in_factor*2, in_factor, bilinear)
        self.outc = OutConv(in_factor, n_classes)

    

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if not self.use_small:

            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
        else:
            x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        # my_out = self.output(logits)
        # return my_out[:,0]
        
if __name__ == "__main__":
#h dim is given by image size and model
    test_model = UNet(1,1, use_small=True, in_factor=64)

    test_input = torch.zeros((            8,
            1,
            50,
            50,))
    my_output = test_model(test_input)
    print(test_model(test_input).shape)
    import matplotlib.pyplot as plt
    plt.imshow(my_output[0,0].detach().cpu().numpy())