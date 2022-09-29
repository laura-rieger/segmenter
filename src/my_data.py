from cmath import phase
import numpy as np
import os
from os.path import join as oj
from skimage import data, io, filters


def load_full(data_path):
    files = os.listdir(data_path)

    im = io.imread(oj(data_path, files[0]))
    return im[:, 0]


def load_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:

        im = io.imread(oj(data_path, file_name))
        # print(im.shape)

        my_data.append(np.asarray(im))
    all_arr = np.asarray(my_data)
    return np.swapaxes(all_arr, 0, 1)


def load_dummy_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:

        im = io.imread(oj(data_path, file_name))
        if im.shape[2] == 3:
            im = np.swapaxes(im, 0, 2)
        print(im.shape)
        imgs = np.vstack(
            [
                im[:, :1024, :1024],
                im[:, :1024, 1024:],
                im[:, 1024:, 1024:],
                im[:, 1024:, :1024],
            ]
        )

        my_data.append(np.asarray(imgs))
    all_arr = np.asarray(my_data)
    return all_arr[0][:, None], all_arr[1]


def load_second(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:

        with open(oj(data_path, files[0]), "rb") as f:

            im = io.imread(oj(data_path, file_name))

            imgs = np.vstack(
                [
                    im[None, :1024, :1024],
                    im[None, :1024, 1024:],
                    im[None, 1024:, 1024:],
                    im[None, 1024:, :1024],
                ]
            )
        my_imgs = np.asarray(imgs)

        my_data.append(my_imgs)
    print(my_data[0].shape)

    my_data[0] = make_classes(my_data[0])
    return my_data


def load_layer_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:  # careful: currently depends on order of files

        im = io.imread(oj(data_path, file_name))
        if im.shape[2] == 3:
            im = np.swapaxes(im, 0, 2)
        imgs = np.vstack(
            [
                im[:, :1024, :1024],
                im[:, :1024, 1024:],
                im[:, 1024:, 1024:],
                im[:, 1024:, :1024],
            ]
        )

        my_imgs = np.asarray(imgs)

        my_data.append(my_imgs)

    # assume that first is x, second y
    if len(my_data[0].shape) < 4:

        my_data[0] = my_data[0][:, None]  # unet expects 4d
    my_data[1] = make_classes(my_data[1])
    return my_data


def make_classes(y):
    y_all = np.zeros_like(y)
    class_vals = np.unique(y)

    num_classes = len(class_vals)
    # print(num_classes)
    my_channels = np.argsort(class_vals)
    for i in range(num_classes):
        y_all[np.where(y == class_vals[my_channels[i]])] = i
    return y_all


def make_dataset(
    x,
    y,
    img_size=25,
    offset=20,
):

    # assume that we have two tif files
    x_list = []
    y_list = []
    img_width = x.shape[-1]

    for idx in range(len(x)):
        # for idx in range(1):

        cur_x, cur_y = 0, 0
        while cur_x <= img_width - img_size:
            cur_y = 0
            while cur_y <= img_width - img_size:

                # here is where you need to
                x_list.append(
                    x[idx, :, cur_x : cur_x + img_size, cur_y : cur_y + img_size]
                )
                y_list.append(
                    y[idx][cur_x : cur_x + img_size, cur_y : cur_y + img_size]
                )
                cur_y += offset
            cur_x += offset
    x_return = np.asarray(x_list).astype(np.float)

    return x_return, np.asarray(y_list)
