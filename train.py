# -*- coding: utf-8 -*-
"""
This file is a modified version based on Sli2Vol.
@author: adlsn
"""
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import os
import numpy as np
from random import shuffle
from tensorboardX import SummaryWriter
import platform
import pickle
import random
import scipy.io as sio
import json

from CorrespondingNet import weight_init

try:
    import matplotlib

    matplotlib.use("TkAgg")
except:
    a = 1

from model import *
from dataset import *


def main():
    # try:
    seed=np.random.randint(0,1000,1)[0]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    datatype = "formal_training"  # TODO: change to "formal_training" for the final training
    interval = 1000000000
    set_size = 2
    batch_size = 16
    in_channels = 48
    R = 7
    params = {
        "batch_size": int(batch_size),
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False,
    }
    max_epochs = 4
    learning_rate = 0.0001
    cycle = False

    metrics_name = [
        "mae",
        # 'ssim',
        # 'profile',
        #                    'consistency',
        #                    'occ',
        "total",
    ]
    loss_weights = {
        "mae": 1,
        # 'profile':1000,
        # 'ssim':1,
        #                    'consistency':1,#0.2
        #                    'occ':10,
        "total": 1,
    }
    running_metrics = {}
    total_metrics = {}
    loss = {}
    for mn in metrics_name:
        running_metrics.update({mn: 0.0})
        total_metrics.update({mn: 0.0})
        loss.update({mn: 0})

    if datatype == "C4KC_KiTS":
        """
        Files-C4KC_KiTS
        """
        from dataset import Dataset_C4KC_KiTS as Dataset

        headfolder = r""

        subfolders = glob.glob(os.path.join(headfolder, "*", "*", "*"))

        folders_training = subfolders[0: int(len(subfolders) * 0.9)]
        folders_validation = subfolders[int(len(subfolders) * 0.9): len(subfolders)]

    elif datatype == "formal_training":
        from dataset import Dataset_PL_Assisted as Dataset

        headfolder = r"data/final_data/TrainingSet/imagesTr"
        subfolders = glob.glob(os.path.join(headfolder, "*"))
        folders_training = subfolders[0: int(len(subfolders) * 0.9)]
        folders_validation = subfolders[int(len(subfolders) * 0.9): len(subfolders)]

    """
    Save path
    """
    path_name = "PL_assisted_model_param/PL_Assisted_REFINE_stream2_data1"

    save_path = os.path.join(os.getcwd(), path_name)

    """
    Model
    """
    saved_model = os.path.join(save_path, "best_model.pth")
    current_model = os.path.join(save_path, "current_model.pth")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = OEG_CFN(
        in_channels=in_channels, is_training=True, R=6
    ).cuda()
    model.apply(weight_init)
    # model.load_state_dict(torch.load("sli2vol_stage_1/best_model.pth"))

    # Loss and optimizer
    patience = 3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5)

    """
    Log
    """
    writer = SummaryWriter(os.path.join(save_path, "log"))

    # Early Stop
    stop_count = 0

    def stop_early(best, current, count):
        stop = False
        if current <= best:
            count = 0
        else:
            count += 1
            if count >= 2 * patience + 1:
                stop = True

        return stop, count

    # Loop over epochs
    best_loss_on_test = np.Infinity
    loss_on_test = {}
    count = 0
    total_count = 0

    """
    Start training
    """
    for epoch in range(max_epochs):

        """
        Training
        """

        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]

        model = model.train()
        for sub in range(0, len(folders_training), interval):
            sub_list = folders_training[
                       sub: sub + min(interval, len(folders_training))
                       ]
            training_set = Dataset(
                sub_list,
                batch_size,
                set_size,
                mode="training_pl_fuse",
                label_mode="Refined_PseudoLabel",  #  two label mode, Pseudo_label and Refined_PseudoLabel
            )
            training_set.shuffle_list()
            training_generator = data.DataLoader(
                training_set, **params, collate_fn=my_collate
            )

            print(path_name)
            i = 0
            for frame1_input, frame2_input, mask1, mask2 in training_generator:

                # Transfer to GPU
                frame1_input = frame1_input.float().cuda()
                frame2_input = frame2_input.float().cuda()
                frame1 = frame1_input.float().cuda()
                frame2 = frame2_input.float().cuda()
                mask1 = mask1.float().cuda()
                mask2 = mask2.float().cuda()

                [frame1_input, frame2_input] = edge_profile(
                    [frame1_input, frame2_input], False, 3, 1
                )
                [mask1, mask2] = edge_profile([mask1, mask2], False, 3, 1)

                b, c, h, w = frame1_input.size()

                optimizer.zero_grad()

                outputs = model(frame1_input, frame2_input, frame1, mask1, mask2)

                outputs = F.interpolate(outputs, (h, w), mode="bilinear")

                loss["mae"] = F.smooth_l1_loss(outputs, frame2, reduction="mean")

                total = 0.0
                for key in loss:
                    if key != "total":
                        total += loss[key] * loss_weights[key]
                loss.update({"total": total})

                i += 1
                if epoch == 0 and sub == 0 and i < 100:
                    for key in loss:
                        print(
                            key,
                            ":",
                            " ",
                            "%.3f" % (loss[key].item() * loss_weights[key]),
                            end="    ",
                        )
                    print("learning rate: %.4e" % current_lr)
                loss["total"].backward()
                optimizer.step()

                for key in loss:
                    running_metrics[key] += loss[key].item() * loss_weights[key]
                    total_metrics[key] += loss[key].item() * loss_weights[key]
                total_count += 1

                # print statistics
                save_freq = 100
                if i % save_freq == 99:  # print every 100 mini-batches
                    print("[%d, %2d, %5d]" % (epoch + 1, sub, i + 1), end=" ")
                    for key in metrics_name:
                        print(
                            key, ": ", "%.3f" % (running_metrics[key] / i), end="    "
                        )
                    print("learning rate: %.4e" % current_lr)

                    # write to log
                    writer.add_scalars("training", running_metrics, count)
                    writer.add_scalar("training/lr", current_lr, count)
                    torch.save(model.state_dict(), saved_model)
                    count += 1

            if i == 0:
                continue

            for key in running_metrics:
                running_metrics[key] /= i
            print("[%d, %2d, %5d]" % (epoch + 1, sub, i + 1), end=" ")
            for key in metrics_name:
                print(key, ": ", "%.3f" % (running_metrics[key]), end="    ")
            print("learning rate: %.4e" % current_lr)

            for key in running_metrics:
                running_metrics[key] = 0.0

        # write to log
        for key in total_metrics:
            total_metrics[key] /= total_count

        for key in total_metrics:
            total_metrics[key] = 0.0
        total_count = 0
        torch.save(model.state_dict(), current_model)


if __name__ == "__main__":
    main()
