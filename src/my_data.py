import numpy as np
import torch
import sys
import pickle as pkl
import os
from os.path import join as oj
from skimage import io

import torch.nn.functional as F
from PIL import Image
from torch.utils.data import  TensorDataset
debug = True

def make_check_folder(intermittent_path, id):
    if not os.path.exists(intermittent_path):
        os.makedirs(intermittent_path)
    if not os.path.exists(oj(intermittent_path, id)):
        os.makedirs(oj(intermittent_path, id))
        os.makedirs(oj(intermittent_path, id, "images"))
        if debug:
            
            os.makedirs(oj(intermittent_path, id, "entropy"))


        os.makedirs(oj(intermittent_path, id, "human_annotated"))
        os.makedirs(oj(intermittent_path, id, "predictions"))
        os.makedirs(oj(intermittent_path, id, "model"))
    return


def load_annotated_imgs(data_path, class_dict):
    # assumes that everything with _0 is training and everything with _1 is testing
    # assumes that there are two folders, predictions and images
    images_folder = oj(data_path, "images")
    annot_folder = oj(data_path, "human_annotated")
    assert len(os.listdir(images_folder)) == len(os.listdir(annot_folder))
    images = []
    annotations = []
    is_test_list = []
  

    for file_name in sorted(os.listdir(images_folder)):

        images.append(io.imread(oj(images_folder, file_name)))
        annotations.append(io.imread(oj(annot_folder, file_name)))
        if "_0" in file_name:
            is_test_list.append(False)
        elif "_1" in file_name:
            is_test_list.append(True)
    print(np.unique(np.asarray(annotations)))
    # make a list of the values in the predictions
    annotations = np.asarray(annotations)
    annotation_vals = np.unique(annotations)
    inverse_class_dict = {v: k for k, v in class_dict.items()}
    list_of_old_vals = sorted(list(inverse_class_dict.keys()))

    if True:     # you want this one normally
        for val in annotation_vals:
            assert val in inverse_class_dict.keys()
        new_annotations = np.zeros_like(annotations)
        for val in annotation_vals:
            new_annotations[annotations == val] = inverse_class_dict[val]

    return_dataset_train = TensorDataset(
        torch.Tensor(np.asarray([x for x, is_test_val in zip(images, is_test_list) if not is_test_val])[:, None]), 
        torch.Tensor(np.asarray([x for x, is_test_val in zip(new_annotations, is_test_list) if not is_test_val])[:, ]))
    return_dataset_test = TensorDataset(
        torch.Tensor(np.asarray([x for x, is_test_val in zip(images, is_test_list) if is_test_val])[:, None]),
        torch.Tensor(np.asarray([x for x, is_test_val in zip(new_annotations, is_test_list) if is_test_val])[:, ]))


    return (return_dataset_train, return_dataset_test)


def save_progress(net, image_idxs, images, folder_path,  args, device, results, class_dict, indicator_list, slice_numbers, save_model = True):
    cur_folder = oj(folder_path, results['file_name'])
    make_check_folder(folder_path, results['file_name'])
    if save_model:

        torch.save(net.state_dict(), oj(cur_folder, "model_state.pt"))
    pkl.dump(image_idxs, open(oj(cur_folder, "image_idxs.pkl"), "wb"))
    #if slice number file already exists, append to it
    if os.path.exists(oj(cur_folder, "slice_numbers.txt")):
        with open(oj(cur_folder, "slice_numbers.txt"), "a") as f:

            for img_idx, slice_number in zip(image_idxs, slice_numbers):
                f.write(str(img_idx) + " " + str(slice_number) + "\n")
    else:
        with open(oj(cur_folder, "slice_numbers.txt"), "w") as f:
            for img_idx, slice_number in zip(image_idxs, slice_numbers):
                f.write(str(img_idx) + " " + str(slice_number) + "\n")


    pkl.dump(results, open(oj(cur_folder, "results.pkl"), "wb"))
    pkl.dump(args, open(oj(cur_folder, "args.pkl"), "wb"))
    pkl.dump(class_dict, open(oj(cur_folder, "class_dict.pkl"), "wb"))
    net.eval()
    with torch.no_grad():
        img_t = torch.Tensor(images).to(device)
        predictions = (net.forward(img_t).argmax(dim=1).detach().cpu().numpy().astype(float) )
        if debug:
            output = F.softmax(net.forward(img_t), dim=1)
            entropy = -torch.sum(output * torch.log(output), dim=1).detach().cpu().numpy()
            entropy = entropy / np.max(entropy)
           
        predictions_classes = np.zeros_like(predictions, dtype=np.uint8)
        for key, val in class_dict.items():
            predictions_classes[predictions == key] = val

    for i in range(len(images)):
        im = Image.fromarray(images[i, 0])
        # starting zeros
        zeros = ''.join(['0' for _ in range(4 - len(str(image_idxs[-1][i])))])

        im.save( oj(cur_folder, "images", zeros + str(image_idxs[-1][i]) + "_" + str(indicator_list[i])+ ".tif"), )
        im = Image.fromarray( predictions_classes[ i, ] )
        im.save( oj(cur_folder, "predictions", zeros + str(image_idxs[-1][i]) + "_" + str(indicator_list[i])+ ".tif"), )
        if debug:
            im = Image.fromarray( entropy[ i, ] )
            im.save( oj(cur_folder, "entropy", zeros + str(image_idxs[-1][i]) + "_" + str(indicator_list[i])+ ".tif"), )

    return


def load_full(data_path):
    files = os.listdir(data_path)

    im = io.imread(oj(data_path, files[0]))
    return im


def load_data(data_path):
    files = os.listdir(data_path)
    my_data = []
    for file_name in files:

        im = io.imread(oj(data_path, file_name))
        # print(im.shape)

        my_data.append(np.asarray(im))
    all_arr = np.asarray(my_data)
    return np.swapaxes(all_arr, 0, 1)


def load_pool_data(data_path, ):

    files = os.listdir(data_path)
    file_name = files[0]
    # XXX this is to not load the entire dataset
    im = io.imread(oj(data_path, file_name))[::50]


    if im.shape[2] == 3: # rgb
        im = np.swapaxes(im, 0, 2)
    imgs = np.vstack(
        [
            im[:, :1024, :1024],
            im[:, :1024, 1024:],
            im[:, 1024:, 1024:],
            im[:, 1024:, :1024],
        ]
    )
    del im

    my_imgs = np.asarray(imgs)

    # my_imgs = my_imgs.astype(np.float)

    if len(my_imgs.shape) < 4:
        my_imgs = my_imgs[:, None]  # unet expects 4d

    return my_imgs




def load_reannotation(data_path, vmax=-1, vmin =-1):

    files = os.listdir(data_path)

    for file_name in files:  # careful: currently depends on order of files
        im = io.imread(oj(data_path, file_name))
        # if im.shape[2] == 3:
        #     im = np.swapaxes(im, 0, 2)
        # imgs = np.vstack( [ im[:, :1024, :1024], im[:, :1024, 1024:], im[:, 1024:, 1024:], im[:, 1024:, :1024], ] )
        my_imgs = np.asarray(im)
        

    my_imgs, num_classes, class_dict = make_classes(my_imgs)
    return my_imgs, num_classes, class_dict, 


def load_layer_data(data_path, vmax=-1, vmin =-1):
    files = sorted(os.listdir(data_path))
    if len(files) < 2:
        return load_pool_data(data_path)
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
    my_data[0] = my_data[0].astype(float)
    # print(my_data[0].dtype)

    # my_data[0] /= my_data[0].max()
    if len(my_data[0].shape) < 4:

        my_data[0] = my_data[0][:, None]  # unet expects 4d
    my_data[1], num_classes, class_dict = make_classes(my_data[1])
    return my_data[0], my_data[1], num_classes, class_dict, 


def make_classes(y):
    y_all = np.zeros_like(y)
    class_vals = np.unique(y)
    # print(class_vals)
    class_vals = class_vals[class_vals != 0]  # unmarked pixels do not count in classes
    num_classes = len(class_vals)
    y_all[y == 0] = 255
    my_channels = np.argsort(class_vals)
    class_dict = {i: class_vals[my_channels[i]] for i in range(num_classes)}
    for i in range(num_classes):
        y_all[np.where(y == class_vals[my_channels[i]])] = i
    return y_all, num_classes, class_dict


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
                    x[idx, :, cur_x: cur_x + img_size, cur_y: cur_y + img_size]
                )
                y_list.append(
                    y[idx][cur_x: cur_x + img_size, cur_y: cur_y + img_size]
                )
                cur_y += offset
            cur_x += offset
    x_return = np.asarray(x_list).astype(float)

    return x_return, np.asarray(y_list)


def make_dataset_single(
    x,
    img_size=25,
    offset=20,
    return_slice_numbers = False
):

    x_list = []
    img_width = x.shape[-1]
    if return_slice_numbers:
        slice_numbers = []

    for idx in range(len(x)):
        # for idx in range(1):

        cur_x, cur_y = 0, 0
        while cur_x <= img_width - img_size:
            cur_y = 0
            while cur_y <= img_width - img_size:

                # here is where you need to
                x_list.append(
                    x[idx, :, cur_x: cur_x + img_size, cur_y: cur_y + img_size]
                )
                if return_slice_numbers:
                    slice_numbers.append(idx)

                cur_y += offset
            cur_x += offset
    x_return = np.asarray(x_list).astype(float)
    if return_slice_numbers:
        return (x_return, np.asarray(slice_numbers))
    return x_return
