from cmath import phase
import numpy as np
import os
from os.path import join as oj
from skimage import data, io, filters
import pickle as pkl

img_width = 2048
radius = 1030
mid_point = int(img_width / 2)
xx, yy = np.mgrid[:img_width, :img_width]

background_color = 2
phase_1 = 67
phase_2 = 224
mean = 115.92
std = 42.25
circle_shape = ((xx - mid_point)**2 +
                (yy - mid_point)**2) < (np.square(radius))


def load_full(data_path):
    files = os.listdir(data_path)

    im = io.imread(oj(data_path, files[0]))
    return im[:, 0]
    return np.asarray(im)


def load_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:

        im = io.imread(oj(data_path, file_name))
        # print(im.shape)

        my_data.append(np.asarray(im))
    all_arr = np.asarray(my_data)
    return np.swapaxes(all_arr, 0, 1)


def load_small_res_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:

        im = io.imread(oj(data_path, file_name))
        if im.shape[2] == 3:
            im = np.swapaxes(im, 0, 2)

        imgs = np.vstack([
            im[:, :1024, :1024], im[:, :1024, 1024:], im[:, 1024:, 1024:],
            im[:, 1024:, :1024]
        ])

        my_data.append(np.asarray(imgs))
    all_arr = np.asarray(my_data)
    return all_arr[0][:, None], all_arr[1]
    print(all_arr.shape)
    return np.swapaxes(all_arr, 0, 1)


def load_layer_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:
        if 'pkl' not in file_name:

            im = io.imread(oj(data_path, file_name))
            if im.shape[2] == 3:
                im = np.swapaxes(im, 0, 2)
            imgs = np.vstack([
                im[:, :1024, :1024], im[:, :1024, 1024:], im[:, 1024:, 1024:],
                im[:, 1024:, :1024]
            ])

        else:
            with open(oj(data_path, files[0]), 'rb') as f:
                im = pkl.load(f)
                imgs = np.vstack([
                    im[:, :, :1024, :1024], im[:, :, :1024, 1024:],
                    im[:, :, 1024:, 1024:], im[:, :, 1024:, :1024]
                ])
                imgs = imgs

        my_data.append(np.asarray(imgs))
    return my_data


def make_dataset(
    x,
    y,
    img_size=25,
    offset=20,
):
    img_width = x.shape[2]

    # shape = imgs[0].shape
    # mid_point = int(img_width / 2)
    # xx, yy = np.mgrid[:img_width, :img_width]

    # convert segmentation to one two three
    y_all = np.zeros_like(y)
    y_all[np.where(y == phase_1)] = 1

    y_all[np.where(y == phase_2)] = 2

    # assume that we have two tif files
    x_list = []
    y_list = []

    for idx in range(len(x)):
        # for idx in range(1):

        cur_x, cur_y = 0, 0
        while (cur_x <= img_width - img_size):
            cur_y = 0
            while (cur_y <= img_width - img_size):

                # here is where you need to
                x_list.append(x[idx, :, cur_x:cur_x + img_size,
                                cur_y:cur_y + img_size])
                y_list.append(y_all[idx][cur_x:cur_x + img_size,
                                         cur_y:cur_y + img_size])
                cur_y += offset
            cur_x += offset
    return (((np.asarray(x_list).astype(np.float)) - mean) /
            std), np.asarray(y_list)
