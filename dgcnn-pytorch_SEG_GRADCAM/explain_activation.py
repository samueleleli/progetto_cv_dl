import os
import argparse

import numpy
import numpy as np
import torch
import torch.nn as nn
from data import S3DIS, ArCH, Sinthcity, ModelNet40, ShapeNetPart
from model import DGCNN_semseg, DGCNN_cls
from torch.utils.data import DataLoader
from gradcam_exp import gradcam

import paramSettings
import configSettings

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy


# os.makedirs("results/activationPlot",exist_ok=True)


def explain_act_semseg(args):
    ###
    if True:
        if True:
            os.makedirs('checkpoints/' + args.exp_name + "/actGradExtractionPlot", exist_ok=True)
            os.makedirs('checkpoints/' + args.exp_name + "/actGradExtractionPlot/actGradExtractionPlotA", exist_ok=True)
            # objs = np.load(configSettings.DATASET_OBJS)
            # labs = np.genfromtxt(configSettings.DATASET_LABS, delimiter=' ').astype("int64")

            total_areas = 9
            args.num_classes = 9
            test_dataset = Sinthcity(partition='test', num_points=args.num_points, test_area=args.test_area)
            test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            preds = np.genfromtxt('checkpoints/' + args.exp_name + "/prediction.txt", delimiter=' ').astype("int64")


            # plot only activation
            i = 0
            idx = np.array(configSettings.BATCH_IDX)
            # for data, gt in test_loader:
            for data, seg, max in test_loader:
                if (i in idx):
                    # a = np.load("results/actGradExtraction/act_conv5_{}.npy".format(idx[i]))
                    a = np.load('checkpoints/' + args.exp_name + "/actGradExtraction/act_conv7_{}.npy".format(i))

                    aM = np.median(a, axis=1)
                    var = aM[0]

                    min_v = np.min(var)
                    max_v = np.max(var)
                    # gt = seg[0] # prende le 4096 etichette associate ai punti del batch i
                    #  = preds[i]

                    data[:, [1, 2]] = data[:, [2, 1]]

                    # varst = (var - min_v) / (max_v - min_v)  # +0.000001)

                    # simmetrizzazione
                    abs_max_v = np.maximum(abs(min_v), abs(max_v))
                    min_v = -abs_max_v
                    max_v = abs_max_v
                    varst = (var - min_v) / (max_v - min_v)  # +0.000001)

                    ply = data  # numpy.stack((data, axis=-1)
                    pcd = o3d.geometry.PointCloud()

                    cmap = plt.cm.get_cmap("jet")
                    varst = cmap(varst)[:, :3]

                    pcd.points = o3d.utility.Vector3dVector(ply[0][:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(varst)

                    o3d.io.write_point_cloud(
                        'checkpoints/' + args.exp_name + "/actGradExtractionPlot/actGradExtractionPlotA/a_median_{}.ply".format(
                            i), pcd)

                    print(i)
                i += 1


class args(object):
    model_path = configSettings.MODEL_PATH  # "models/model.cls.1024.t7" #
    model = configSettings.MODEL
    k = 20  # non utilizzato in segmentazione
    emb_dims = 1024  # non utilizzato in segmentazione
    dropout = 0  # 0.5   # non utilizzato in segmentazione
    # num_classes = configSettings.OUTPUT_CHANNELS --- viene settato nel codice del test
    no_cuda = paramSettings.NO_CUDA
    output_channels = configSettings.OUTPUT_CHANNELS  # 15 # 40
    # aggiunti per segmentazione
    exp_name = configSettings.EXP_DIR
    dataset = configSettings.TEST_DATASET
    test_area = configSettings.TEST_AREA
    test_batch_size = configSettings.TEST_BATCH_SIZE
    model_root = configSettings.MODEL_ROOT
    parallel = configSettings.PARALLEL
    num_points = configSettings.NUM_POINTS
    seed = configSettings.SEED
    # extract = configSettings.EXTRACT
    # eval = configSettings.EVAL
    lr = configSettings.LR
    use_sgd = configSettings.USE_SGD
    scheduler = configSettings.SCHEDULER
    momentum = configSettings.MOMENTUM


explain_act_semseg(args)