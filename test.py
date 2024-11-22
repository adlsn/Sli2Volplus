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
import cv2
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import dice, jaccard
import imageio
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    binary_erosion,
    distance_transform_edt,
)
from skimage.measure import label
from skimage.filters.rank import median
from skimage.morphology import disk, convex_hull_image
from scipy.ndimage import generic_filter

# from sklearn import svm
# import pydensecrf.densecrf as dcrf
# from evaluation_metrics import *
from skimage.segmentation import slic
from itertools import zip_longest
import time
import nibabel as nib

from pl_assisted_model import *  # VGG, VGG_uncertainty, VGG_uncertainty_constant,VGG_uncertainty_absolute, VGG_whole_average, VGG_whole_average_v2, VGG_whole_attention, VGG_whole_attention_v2,VGG_whole_attention_v3, VGG_intertwined_attention, VGG_intertwined_attention_v2, VGG_full_map, make_layers,make_layers_non_local,make_layers_group_norm, weight_init
from dataset import *
import argparse
from monai.metrics import DiceMetric

from model import OEG_CFN

try:
    import matplotlib

    matplotlib.use("TkAgg")
except:
    a = 1


def verification_module(mask_img, mask_collection, img, img_collection, template):
    def getLargestCC(segmentation):
        labels = label(segmentation)
        if labels.max() == 0:
            return segmentation
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return (largestCC > 0).astype(int)

    def refine(segmentation, original):
        chull = convex_hull_image(segmentation)
        return segmentation  # chull#*original

    #    print(mask_img.shape, mask_collection.shape, img.shape, img_collection.shape)
    """
    mask_img - (h, w)
    mask_collection - (f, h, w)
    img - (c, h, w)
    img_collection - (f, c, h, w)
    """
    num_class = int(np.max(mask_collection[-1]))
    mask_refine = np.zeros(mask_img.shape)

    for nc in range(1, num_class + 1, 1):
        mask_history = (mask_collection[-1] == nc).astype(float)
        mask_img_temp = (mask_img == nc).astype(float)

        ###
        mask_neg = binary_dilation(mask_history, iterations=5) - mask_history
        mask_neg = np.tile(np.expand_dims(mask_neg, 0), (img.shape[0], 1, 1))
        mask_pos = np.tile(
            np.expand_dims(mask_history, 0), (img.shape[0], 1, 1)
        )  # c, h, w

        feature_positive = np.ones(img.shape) * np.tile(
            np.array(
                [
                    np.mean(img_collection[-1, i][mask_pos[i] == 1])
                    for i in range(mask_pos.shape[0])
                ]
            )[:, np.newaxis, np.newaxis],
            (1, img.shape[-2], img.shape[-1]),
        )
        feature_negative = np.ones(img.shape) * np.tile(
            np.array(
                [
                    np.mean(img_collection[-1, i][mask_neg[i] == 1])
                    for i in range(mask_neg.shape[0])
                ]
            )[:, np.newaxis, np.newaxis],
            (1, img.shape[-2], img.shape[-1]),
        )

        ###

        feature_positive = np.sum((img - feature_positive) ** 2, axis=0)
        feature_negative = np.sum((img - feature_negative) ** 2, axis=0)

        positive_likelihood = np.zeros(feature_positive.shape)
        positive_likelihood[feature_positive < feature_negative] = 1
        positive_likelihood = (
                positive_likelihood * mask_img_temp
        )  # + (binary_erosion(mask_img_temp, iterations=5))
        positive_likelihood[positive_likelihood > 0] = 1

        #    positive_likelihood = np.exp(img*feature_positive)/(np.exp(img*feature_positive)+np.exp(img*feature_negative))
        #    positive_likelihood[positive_likelihood>0.5]=1
        #    positive_likelihood[positive_likelihood<=0.5]=0
        #    positive_likelihood = positive_likelihood*mask_img_temp

        positive_likelihood[~np.isfinite(positive_likelihood)] = 0
        # positive_likelihood = getLargestCC(positive_likelihood)
        # positive_likelihood=refine(positive_likelihood, mask_img_temp)

        positive_likelihood = binary_fill_holes(positive_likelihood).astype(int)

        positive_likelihood = binary_dilation(positive_likelihood, iterations=2)
        positive_likelihood = binary_erosion(positive_likelihood, iterations=2)

        #    print(np.sum(positive_likelihood), np.unique(positive_likelihood), positive_likelihood.shape)
        #    feature_negative=feature_negative[132323]

        positive_likelihood = binary_fill_holes(positive_likelihood).astype(int)

        mask_refine[positive_likelihood > 0] = nc

    return mask_refine, template


def main():
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--output_pth", type=str, default="", required=True, help="output path"
    )
    parse.add_argument(
        "--test_data_pth", type=str, default="", required=True, help="test data path"
    )
    parse.add_argument(
        "--model_pth", type=str, default="", required=True, help="model path"
    )
    args = parse.parse_args()

    plt.close("all")
    seed = np.randint(0, 1000, 1)[0]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #    torch.cuda.set_device(0)

    """
    Parameters
    """
    datatype = "formal_training"
    interval = 100
    batch_size = 12
    set_size = 2
    in_channels = 48
    R = 7
    params = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": 0,
        "drop_last": False,
    }
    cycle = False

    if datatype == "decathon_liver":
        """
        Files-decathon_liver
        """
        from dataset import Dataset_test_decathon_liver as Dataset

        headfolder = "data/Task03_Liver/labelsTr"
        subfolders = os.listdir(headfolder)  # nii.gz files

        subfolders = [os.path.join(headfolder, s) for s in subfolders]

        folders_training = subfolders[0: int(len(subfolders) * 0.8)]
        folders_validation = subfolders[int(len(subfolders) * 0): len(subfolders)]

    elif datatype == "formal_training":
        from dataset import Dataset_test_PL_Assisted as Dataset

        headfolder = args.test_data_pth
        subfolders = glob.glob(os.path.join(headfolder, "*"))
        folders_training = subfolders[0: int(len(subfolders) * 0.9)]
        folders_validation = subfolders[int(len(subfolders) * 0): len(subfolders)]

    """
    Model
    """
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    mother_path = args.model_pth

    model_path = os.path.join(mother_path, "best_model.pth")
    video_path = mother_path

    model = OEG_CFN(
        in_channels=in_channels, is_training=False, R=6
    ).cuda()
    model.load_state_dict(torch.load(model_path))

    """
    testing
    """
    result_overall = {}
    model = model.eval()
    pos_name = [
        #            0,
        #            25,
        #            50,
        #            75,
        #            100,
        200
    ]
    with torch.set_grad_enabled(False):
        result_model = []
        result_repeat = []
        result_group = {}
        decay = []
        for img_num, folder in enumerate(folders_validation):
            # if img_num<7:
            #     continue
            validation_set = Dataset([folder], 1, set_size, mode="validation")
            validation_generator = data.DataLoader(
                validation_set, **params, collate_fn=my_collate
            )
            for (
                    img_vol_norm,
                    mask_truth,
                    img_vol,
                    pos_max,
                    pos_start,
                    pos_end,
                    file_name,
                    pseudo_vol,
            ) in validation_generator:
                file_name = file_name[0]
                pos_max = (
                    pos_max.squeeze().detach().cpu().numpy().item()
                )  # -10#+random.randint(-5,5)
                pos_start = pos_start.squeeze().detach().cpu().numpy().item()
                pos_end = pos_end.squeeze().detach().cpu().numpy().item()

                mask_truth = mask_truth.squeeze().detach().cpu().numpy()
                mask_truth[mask_truth > 0] = 1
                pseudo_vol[pseudo_vol > 0] = 1

                for pos_num, pos_current in enumerate(
                        [
                            #                                                   pos_start+1,
                            #                                                   int(pos_start+0.25*(pos_end-pos_start)),
                            #                                                   int(pos_start+0.5*(pos_end-pos_start)),
                            #                                                   int(pos_start+0.75*(pos_end-pos_start)),
                            #                                                   pos_end-1,
                            pos_max,
                        ]
                ):
                    mask_pred = mask_truth.copy()
                    mask_pred *= 0
                    mask_pred[pos_current] = mask_truth[pos_current].copy()

                    time_count = []
                    """
                    backward
                    """
                    template = []
                    img_collection = (img_vol.squeeze().detach().cpu().numpy())[
                                     pos_current: pos_current + 1
                                     ][np.newaxis]
                    mask_collection = mask_truth[pos_current: pos_current + 1].copy()
                    feat_collection = np.ones([])
                    temp = []
                    for i in range(pos_current, 0, -1):
                        # Transfer to GPU
                        frame1_input = img_vol_norm[:, i: i + 1].float().cuda()
                        frame2_input = img_vol_norm[:, i - 1: i].float().cuda()

                        frame1 = img_vol[:, i: i + 1].float().cuda()
                        frame2 = img_vol[:, i - 1: i].float().cuda()

                        mask_1 = pseudo_vol[:, i: i + 1].float().cuda()
                        mask_2 = pseudo_vol[:, i - 1: i].float().cuda()

                        start_time = time.time()

                        [frame1_input, frame2_input] = edge_profile(
                            [frame1_input, frame2_input], False, 3, 1
                        )
                        [mask1_input, mask2_input] = edge_profile(
                            [mask_1, mask_2], False, 3, 1
                        )

                        mask1 = (
                            torch.from_numpy(mask_pred[i: i + 1])
                            .unsqueeze(0)
                            .float()
                            .cuda()
                        )

                        b, c, h, w = frame1_input.size()

                        _output = model(
                            frame1_input, frame2_input, mask1, mask1_input, mask2_input
                        )
                        _output = F.interpolate(_output, (h, w), mode="bilinear")
                        output = torch.argmax(_output, 1, keepdim=True).float()

                        output = output.squeeze().detach().cpu().numpy()

                        output, template = verification_module(
                            output,
                            mask_collection,
                            (img_vol.squeeze().detach().cpu().numpy())[i - 1: i],
                            img_collection,
                            template,
                        )

                        time_count.append(time.time() - start_time)

                        mask_pred[i - 1] = output

                        mask_collection = np.concatenate(
                            (mask_collection, output[np.newaxis]), axis=0
                        )
                        img_collection = np.concatenate(
                            (
                                img_collection,
                                (img_vol.squeeze().detach().cpu().numpy())[
                                np.newaxis, i - 1: i
                                ],
                            ),
                            axis=0,
                        )
                        dice_metric(mask_pred, mask_truth)
                        dice_score = dice_metric.aggregate().item()
                        dice_metric.reset()
                        temp.append(
                            dice_score
                        )

                        if np.max(output) == 0:
                            break

                    decay.append(temp)

                    """
                    forward
                    """
                    template = []
                    img_collection = (img_vol.squeeze().detach().cpu().numpy())[
                                     pos_current: pos_current + 1
                                     ][np.newaxis]
                    mask_collection = mask_truth[pos_current: pos_current + 1].copy()
                    feat_collection = np.ones([])
                    temp = []
                    for i in range(pos_current, mask_pred.shape[0] - 1, 1):
                        # Transfer to GPU
                        frame1_input = img_vol_norm[:, i: i + 1].float().cuda()
                        frame2_input = img_vol_norm[:, i + 1: i + 2].float().cuda()

                        frame1 = img_vol[:, i: i + 1].float().cuda()
                        frame2 = img_vol[:, i + 1: i + 2].float().cuda()

                        mask_1 = pseudo_vol[:, i: i + 1].float().cuda()
                        mask_2 = pseudo_vol[:, i + 1: i + 2].float().cuda()

                        start_time = time.time()

                        [frame1_input, frame2_input] = edge_profile(
                            [frame1_input, frame2_input], False, 3, 1
                        )
                        [mask1_input, mask2_input] = edge_profile(
                            [mask_1, mask_2], False, 3, 1
                        )

                        mask1 = (
                            torch.from_numpy(mask_pred[i: i + 1])
                            .unsqueeze(0)
                            .float()
                            .cuda()
                        )

                        b, c, h, w = frame1_input.size()

                        _output = model(
                            frame1_input, frame2_input, mask1, mask1_input, mask2_input
                        )
                        _output = F.interpolate(_output, (h, w), mode="bilinear")
                        output = torch.argmax(_output, 1, keepdim=True).float()

                        output = output.squeeze().detach().cpu().numpy()

                        output, template = verification_module(
                            output,
                            mask_collection,
                            (img_vol.squeeze().detach().cpu().numpy())[i + 1: i + 2],
                            img_collection,
                            template,
                        )

                        time_count.append(time.time() - start_time)

                        mask_pred[i + 1] = output

                        mask_collection = np.concatenate(
                            (mask_collection, output[np.newaxis]), axis=0
                        )
                        img_collection = np.concatenate(
                            (
                                img_collection,
                                (img_vol.squeeze().detach().cpu().numpy())[
                                np.newaxis, i + 1: i + 2
                                ],
                            ),
                            axis=0,
                        )
                        dice_metric(mask_pred, mask_truth)
                        dice_score = dice_metric.aggregate().item()
                        dice_metric.reset()
                        temp.append(
                            dice_score
                        )

                        if np.max(output) == 0:
                            break

                    decay.append(temp)
                    img_vol = img_vol.squeeze().detach().cpu().numpy()
                    """
                    Results
                    """
                    print(file_name)
                    result_temp = {}
                    for i in range(1, int(np.max(mask_truth)) + 1):
                        dice_metric(mask_pred == i, mask_truth == i)
                        dice_score = dice_metric.aggregate().item()
                        dice_metric.reset()
                        result_temp.update(
                            {
                                i: dice_score
                            }
                        )
                        print(
                            i,
                            dice_score
                        )

                    result_overall.update({file_name: result_temp})

                    "Save prediction"

                    mask_truth = mask_truth.transpose(1, 2, 0)
                    mask_pred = mask_pred.transpose(1, 2, 0)

                    mask_truth = nib.Nifti1Image(
                        mask_truth.astype(np.int8), np.eye(4)
                    )  # Save axis for data (just identity)
                    mask_truth.header.get_xyzt_units()

                    file_pth = pathlib.Path(
                        file_name.replace("labelsTr", args.output_pth)
                    )
                    file_dir = file_pth.parent
                    if file_dir.exists() == False:
                        file_dir.mkdir(parents=True)

                    mask_truth.to_filename(
                        file_name.replace(
                            "labelsTr", args.output_pth
                        )
                    )  # Save as NiBabel file

                    mask_pred = nib.Nifti1Image(
                        mask_pred.astype(np.int8), np.eye(4)
                    )  # Save axis for data (just identity)
                    mask_pred.header.get_xyzt_units()
                    mask_pred.to_filename(
                        file_name.replace("labelsTr", args.output_pth).replace(
                            ".nii.gz", "_pred.nii.gz"
                        )
                    )  # Save as NiBabel file

        """
        Print result overall
        """
        mean_store = []
        std_store = []
        num_label = 0
        for file_name in result_overall:
            if len(result_overall[file_name]) > num_label:
                num_label = len(result_overall[file_name])
        for label_type in range(1, num_label + 1, 1):
            temp = []
            print(label_type)
            for file_name in result_overall:
                try:
                    temp.append(result_overall[file_name][label_type])
                except:
                    a = 1
                # print(file_name)
            #            [print(temp[i]) for i in range(len(temp))]
            mean_store.append(np.mean(temp))
            std_store.append(np.std(temp))
        #            print('mean:', np.mean(temp))
        #            print('std:', np.std(temp))
        print("mean:")
        [print(mean_store[i]) for i in range(len(mean_store))]
        print(np.mean(mean_store))
        print("std:")
        [print(std_store[i]) for i in range(len(std_store))]


if __name__ == "__main__":
    main()
