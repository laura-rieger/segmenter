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
def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument( "--model_name", type=str, default="9480060078", )
    parser.add_argument("--input_folder", "-i", type=str, 
                        default="C:\\Users\\lauri\\Documents\\GitHub\\segmenter\\data\\predict_folder\\LNO.tif") 
    parser.add_argument("--result_folder", 
                        "-r", type=str, default="C:\\Users\\lauri\\Documents\\GitHub\\segmenter\\data\\result_folder") 
    
    return parser.parse_args()
def run(results, config ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make model
    net = UNet( n_channels=1, n_classes=results["num_classes"] ).to(device=device)
    #check if there is a model
    if os.path.exists(oj(config["PATHS"]["model_path"],results['file_name'] + ".pt")):
        net.load_state_dict(torch.load(oj(config["PATHS"]["model_path"],results['file_name'] + ".pt")))
    else:
        net.load_state_dict(torch.load(oj(config["PATHS"]["model_path"],results['file_name'], "model_state.pt")))
    net.eval()

    # load data
    im = io.imread(results['input_folder'])
    #preprocess
    im = im.astype(np.float32)
    im = (im - results['data_min']) / (results['data_max'] - results['data_min'])

    #assume that it can handle one image at a time
    for k in tqdm(range(len(im))):
        cur_input = im[k]
        #divide image into parts

        height, width = cur_input.shape
        complete_output = np.zeros((height, width))
        for i in range(0,4):
            for j in range(0,4):
                cur_cur_input = cur_input[i * height//4:(i+1) * height//4, j * width//4:(j+1) * width//4]
                cur_cur_input = torch.from_numpy(cur_cur_input).unsqueeze(0).unsqueeze(0).to(device=device)
                cur_output = net(cur_cur_input)
                cur_output = cur_output.squeeze(0).argmax(dim=0).cpu().detach().numpy()
                complete_output[i * height//4:(i+1) * height//4, j * width//4:(j+1) * width//4] =cur_output
             

        #save output
        #fill up with zeros
        complete_output_final = np.zeros_like(complete_output)
        for val in results['class_dict']:
            complete_output_final[complete_output == val] = results['class_dict'][val]  
        cur_image = Image.fromarray(complete_output_final.astype(np.uint8))
        filename = ''.join(['0' for _ in range(5-len(str(k)))]) + str(k) + ".tif"
        full_filename = oj(results['result_folder'],filename)
   
        cur_image.save(full_filename)


     





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
        print(arg)   
        results[str(arg)] = getattr(args, arg)

    
    run( results=results, config=config )