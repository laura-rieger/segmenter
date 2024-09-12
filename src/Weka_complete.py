import os
import sys 
from torch.nn import functional as F
import configparser
import pandas as pd 

import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, "../src")
import configparser
import pandas as pd 

from utils.dice_score import multiclass_dice_coeff
import matplotlib
from os.path import join as oj
sys.path.insert(0, "../src")
from unet import UNet
import my_data
import pickle as pkl
pd.set_option('display.float_format', lambda x: '%.3f' % x)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
config = configparser.ConfigParser()
config.read('../config.ini');
data_path =config['DATASET']['data_path'] 
fig_path =config['PATHS']['figure_path'] 
import tifffile as tiff
import numpy as np

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

x, y, num_classes, class_dict = my_data.load_layer_data(
                        oj(config["DATASET"]["data_path"], 'lno')
                    )

x_try = x[0]
y_try = y[0]

input_file = oj(data_path, 'lno_feature_stack','feature-stack0006.tif') 

    # Read the image stack
with tiff.TiffFile(input_file) as tif:
    image_stack_test = tif.asarray()

list_of_training_vectors = []
list_of_training_labels = []
x, y, num_classes, class_dict = my_data.load_layer_data(
                            oj(config["DATASET"]["data_path"], 'lno'))
x_half, y_half, _, _ = my_data.load_layer_data(
                            oj(config["DATASET"]["data_path"], 'lno_halfHour'), )

num = 400
                        
for i in tqdm(range(5)):

    input_file = oj(data_path, 'lno_feature_stack','feature-stack000{}.tif'.format(i+1))    
    # Read the image stack
    with tiff.TiffFile(input_file) as tif:
        image_stack = tif.asarray()
    
    h, w =int( np.random.uniform(0, 2048-num) ), int(np.random.uniform(0, 2048-num))
    image_stack = image_stack[:, h:h+num, w:w+num]
    # break
    image_stack = image_stack.reshape(76,num*num).T
    my_vec = y[i, h:h+num, w:w+num].reshape(-1)
    black_out = np.random.uniform(0, 1, len(image_stack)) < 1.0
    image_stack = image_stack[black_out]
    my_vec = my_vec[black_out]


    list_of_training_labels.append(my_vec)
    list_of_training_vectors.append(image_stack)



         
for i in tqdm(range(4)):

    input_file = oj(data_path, 'lno_feature_stack_half','feature-stack000{}.tif'.format(i+1))    
    # Read the image stack
    with tiff.TiffFile(input_file) as tif:
        image_stack = tif.asarray()
    
    
    # break
    image_stack = image_stack.reshape(76,-1).T
    my_vec = y_half[i].reshape(-1)
    image_stack = image_stack[my_vec != 255]
    my_vec = my_vec[my_vec != 255]

    list_of_training_labels.append(my_vec)
    list_of_training_vectors.append(image_stack)

list_of_training_vectors = np.concatenate(list_of_training_vectors)
list_of_training_labels = np.concatenate(list_of_training_labels)


list_of_results = []

for i in range(5):
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,max_features = 2, random_state= i  )
    clf = clf.fit(list_of_training_vectors, list_of_training_labels)
    y_pred = clf.predict(image_stack_test.reshape(76,-1).T)
    y_pred = y_pred.reshape(2048,2048)
    

    y_pred_one_hot = torch.nn.functional.one_hot(torch.Tensor(y_pred[None,:]).to(torch.int64), 
                                                num_classes=3).permute(0, 3, 1, 2).squeeze()[None, :]
    dice_all = multiclass_dice_coeff(y_pred_one_hot.float(), 
                        torch.Tensor(y[5][None,:]), 
                        num_classes=3,separated_up = True).item()

    dice = multiclass_dice_coeff(y_pred_one_hot.float(), 
                        torch.Tensor(y[5][None,:]), 
                        num_classes=3,separated_up = False).item()
    list_of_results.append([ dice_all, dice])
print("Done")
with open( 'random_forest_classifier_approx_complete.pkl', 'wb') as f:
    pkl.dump(list_of_results, f)

