import argparse
import configparser
import os
from PIL import Image
from os.path import join as oj
import torch
torch.backends.cudnn.deterministic = True
from tqdm import tqdm
from skimage import io
import pickle as pkl
import numpy as np
from unet import UNet


import torch.nn.functional as F
def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument( "--model_name", type=str, default="4352513863", )
    parser.add_argument("--input_folder", "-i", type=str, 
                        default="C:\\Users\\lauri\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\GitHub\\segmenter\\data\\predict_folder\\LNO.tif") 
    parser.add_argument("--result_folder", 
                        "-r", type=str, default="C:\\Users\\lauri\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\GitHub\\segmenter\\data\\result_folder") 
    # parser step size
    parser.add_argument("--step_size", "-s", type=int, 
                        default=128 ) 
    
    return parser.parse_args()


def get_patch_multiplier(patch_size):
    x, y = np.meshgrid(np.arange(patch_size), np.arange(patch_size))

    distance_from_middle = np.sqrt((x - patch_size / 2) ** 2 + (y - patch_size / 2) ** 2) 
    distance_from_middle = distance_from_middle / distance_from_middle.max()
    

    distance_from_middle = 1- distance_from_middle
    distance_from_middle = distance_from_middle +.01
    return distance_from_middle 
def run(net, input_imgs, data_min, data_max, step_size, num_classes, use_orig_values=True, return_whole = False ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.eval()

    im = input_imgs.squeeze()
    if len(im.shape) == 2:
        im = im[None, :]
    #preprocess

    #assume that it can handle one image at a time
    output_imgs = []
    with torch.no_grad():
        for k in range(len(im)):
            cur_input = (np.copy(im[k]).astype(np.float32)- data_min) / (data_max - data_min)
        
            #divide image into parts

            height, width = cur_input.shape
            
            patch_size = height // 4
            # ensure that img size is divisible by step size
            assert height % step_size == 0 and width % step_size == 0

            # Create a meshgrid representing the coordinates of each pixel



            complete_output = np.zeros(( num_classes, height, width,))
       
            for i in tqdm(range(0, height//step_size)):
                for j in range(0, width//step_size):
                    
                    start_height, end_height = i * step_size, i * step_size + patch_size
                    start_width, end_width = j * step_size, j * step_size + patch_size

                    cur_cur_input = cur_input[start_height:end_height, start_width:end_width]

                    
                    cur_cur_input = torch.from_numpy(cur_cur_input).unsqueeze(0).unsqueeze(0).to(device=device)
                    cur_output = F.softmax(net(cur_cur_input), dim=1)
                    cur_output = net(cur_cur_input)
                    if cur_output.shape[2] != patch_size or cur_output.shape[3] != patch_size:
                        break

                    factor = get_patch_multiplier(patch_size)
                    cur_output = cur_output.squeeze().cpu().detach().numpy() * factor[None, :]
                

                    
                    complete_output[:, start_height:end_height, start_width:end_width] +=cur_output
            if return_whole:
                return complete_output
            complete_output = np.argmax(complete_output, axis=0)

            complete_output_final = np.zeros_like(complete_output)
            if not use_orig_values:
                for val in results['class_dict']:
                    complete_output_final[complete_output == val] = results['class_dict'][val]  
            else:
                complete_output_final = complete_output


            # output
            cur_image = Image.fromarray(complete_output_final.astype(np.uint8))
            output_imgs.append(cur_image)
       
    return output_imgs

            


     





if __name__ == "__main__":
    args = get_args()
    config = configparser.ConfigParser()
    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")

     
    # assume that there is a pkl with the results in this folder
    # check if the name isa folder
    if os.path.isdir(oj(config["PATHS"]["model_path"], args.model_name)):
        results = pkl.load(open(oj(config["PATHS"]["model_path"], args.model_name, "results.pkl"), "rb"))
    else:
        results = pkl.load(open(oj(config["PATHS"]["model_path"], args.model_name + ".pkl"), "rb"))
    for arg in vars(args):  

        results[str(arg)] = getattr(args, arg)



    # make model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet( n_channels=1, n_classes=results["num_classes"] ).to(device=device)
    #check if there is a model
    if os.path.exists(oj(config["PATHS"]["model_path"],results['file_name'] + ".pt")):
        net.load_state_dict(torch.load(oj(config["PATHS"]["model_path"],results['file_name'] + ".pt")))
    else:
        net.load_state_dict(torch.load(oj(config["PATHS"]["model_path"],results['file_name'], "model_state.pt")))

    data_min, data_max = results['data_min'], results['data_max']
    input_imgs = io.imread(args.input_folder)

    output_imgs = run(net, input_imgs, data_min, data_max, args.step_size, results['num_classes'], use_orig_values=False )
    for k in range(len(output_imgs)):
        


        filename = ''.join(['0' for _ in range(5-len(str(k)))]) + str(k) + ".tif"
        full_filename = oj(results['result_folder'],filename)
    
        output_imgs[k].save(full_filename)