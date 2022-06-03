import numpy as np
import os
from PIL import Image
from os.path import join as oj


def load_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:
        im = Image.open(oj(data_path, file_name))
        my_data.append(np.asarray(im))
    return my_data


def make_rudimentary_dataset(imgs, img_size=25, offset=20, radius=1030):
    shape = imgs[0].shape
    mid_point = int(shape[0] / 2)
    xx, yy = np.mgrid[:shape[0], :shape[1]]
    circle_shape = ((xx - mid_point)**2 +
                    (yy - mid_point)**2) < (np.square(radius))

    x_list = []
    y_list = []
    for cur_img in imgs:
        cur_x, cur_y = 0, 0
        while (cur_x < shape[0] - img_size):
            cur_y = 0
            while (cur_y < shape[0] - img_size):
                # here is where you need to
                x_list.append(cur_img[cur_x:cur_x + img_size,
                                      cur_y:cur_y + img_size])
                y_list.append(
                    (cur_img[cur_x:cur_x + img_size, cur_y:cur_y + img_size] >
                     150, circle_shape[cur_x:cur_x + img_size,
                                       cur_y:cur_y + img_size]))
                cur_y += offset
            cur_x += offset
    return ((np.asarray(x_list).astype(np.int16) - 128) /
            128), np.asarray(y_list)
