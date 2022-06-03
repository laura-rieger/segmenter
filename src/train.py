import os
import argparse
import configparser
import pickle as pkl

from argparse import ArgumentParser
from copy import deepcopy
from os.path import join as oj
from matplotlib.style import use
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import models
import unet_model
import my_data
from sklearn.model_selection import train_test_split

cuda = torch.cuda.is_available()

device = torch.device("cuda")


def get_args():
    parser = ArgumentParser(description="Template")
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--experiment_name", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)

    ret_args = parser.parse_args()
    return ret_args


args = get_args()

config = configparser.ConfigParser()
config.read("../config.ini")
#%%

save_path = config["PATHS"]["model_path"]
if not os.path.exists(save_path):
    os.makedirs(save_path)

x, y = my_data.make_rudimentary_dataset(
    my_data.load_data(config['DATASET']['data_path']),
    img_size=25,
    offset=25,
)
x = x[:, None]

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.15,
                                                    random_state=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.18, random_state=1)  # 0.25 x 0.8 = 0.2

torch.manual_seed(args.seed)
train_dataset = TensorDataset(
    *[torch.Tensor(input) for input in [X_train, y_train]])
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)

val_dataset = TensorDataset(
    *[torch.Tensor(input) for input in [
        X_val,
        y_val,
    ]])
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

#%%

input_dim = X_train.shape[
    2]  # Number of input features (e.g. discharge capacity)

model = unet_model.UNet(1, 1, use_small=True).to(device)

optimizer = optim.Adam(model.parameters(), )
loss_function = torch.nn.BCELoss(reduction='none')
training_loss = []
validation_loss = []

best_val_loss = 500000

cur_patience = 0
max_patience = 5
patience_delta = 0.0
best_weights = None

#%%

for epoch in range(args.num_epochs):

    model.train()
    tr_loss = 0

    for batch_idx, (
            input_data,
            y_hat,
    ) in enumerate(train_loader):

        input_data = input_data.to(device)
        y_hat = y_hat.to(device)
        optimizer.zero_grad()
        y_pred = model(input_data, )

        # loss
        loss = loss_function(y_pred[:, 0], y_hat[:, 0])
        torch.masked_select()

        (loss).backward()
        tr_loss += loss.item()
        optimizer.step()

    tr_loss /= len(train_loader.dataset)
    training_loss.append(tr_loss)

    model.eval()
    val_loss = 0
    val_loss_state = 0
    val_loss_lifetime = 0

    with torch.no_grad():
        for batch_idx, (
                input_data,
                y_hat,
        ) in enumerate(val_loader):

            input_data = input_data.to(device)
            supp_data = supp_data.to(device)
            y_hat = y_hat.to(device)

            y_pred = model(input_data)

            loss_state = loss_function(y_hat, y_pred)
            loss = loss_state

            val_loss += loss.item()
            val_loss_state += loss_state.item()

    val_loss /= len(val_loader.dataset)
    val_loss_state /= len(val_loader.dataset)

    val_loss_lifetime /= len(val_loader.dataset)
    validation_loss.append(val_loss)

    print("Epoch: %d, Training loss: %1.5f, Validation loss: %1.5f, " % (
        epoch + 1,
        tr_loss,
        val_loss,
    ))

    if val_loss + patience_delta < best_val_loss:
        best_weights = deepcopy(model.state_dict())
        cur_patience = 0
        best_val_loss = val_loss
    else:
        cur_patience += 1
    if cur_patience > max_patience:
        break

#%%

np.random.seed()
file_name = "".join([str(np.random.choice(10)) for x in range(10)])

results = {}
for arg in vars(args):
    if arg != "save_path":
        results[str(arg)] = getattr(args, arg)

results["train_losses"] = training_loss
results["val_losses"] = validation_loss

model.load_state_dict(best_weights)
model.eval()

results["file_name"] = file_name
results["best_val_loss"] = best_val_loss

pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

torch.save(model.state_dict(), oj(save_path, file_name + ".pt"))
print("Saved")
