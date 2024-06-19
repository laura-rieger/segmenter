from typing import Tuple
import os
from os.path import join as oj
import numpy as np
from PIL import Image
import configparser
def main(foldername: str = '5570988367', pool_name_annotated: str = 'lno_pq_annotated', image_size: int =128,   **kwargs) -> Tuple[bool, dict]:
    import my_data
    config = configparser.ConfigParser()

    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")
    tot_folder_path = oj(config["PATHS"]["pq_progress_results"], foldername)

    # load file
    slice_info_path = oj(tot_folder_path, 'slice_numbers.txt')
    # read lines
    with open(slice_info_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = int(lines[i].split(' ')[0])
    # get
    # get names of the images in the images folder
    image_folder = oj(tot_folder_path, 'images')
    predictions_folder = oj(tot_folder_path, 'human_annotated')
    image_files = os.listdir(image_folder)

    # also open class_idct.pkl with pickle
    # and get the class names
    import pickle as pkl
    with open(oj(tot_folder_path, 'class_dict.pkl'), 'rb') as f:
        class_dict = pkl.load(f)


    # print(lines)

    x_pool, y_pool, _, _ = my_data.load_layer_data( oj(config["DATASET"]["data_path"], pool_name_annotated) )
    # MAX min
    _, y_pool_all = my_data.make_dataset( x_pool[:-1], y_pool[:-1], img_size=image_size, offset=image_size, )


    print(x_pool.shape)
    
    for i in range(len(lines)):
        predictions_classes = np.zeros_like(y_pool_all[lines[i]], dtype=np.uint8)
        for key, val in class_dict.items():
            predictions_classes[y_pool_all[lines[i]] == key] = val
        im = Image.fromarray(predictions_classes)
        # starting zeros
        # find name
        my_name = [x for x in image_files if str(lines[i]) in x][0] # lmao but it works

        im.save( oj(predictions_folder,my_name, ))
    return True, {'progress_folder' : foldername}


    # next load the names 
if __name__ == '__main__':
    main()