import os
import argparse

import numpy
import numpy as np
import torch
#from torch._C import float32
import torch.nn as nn
from data import S3DIS, ArCH, Sinthcity, ModelNet40, ShapeNetPart
from model import DGCNN_semseg, DGCNN_cls
from torch.utils.data import DataLoader
from gradcam_exp import gradcam

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

import paramSettings
import configSettings
import itertools



# os.makedirs("results/gradcamPlot",exist_ok=True)

def explain_gradcam_semseg(args):
    ###
    if True:
        if True:
            os.makedirs('checkpoints/' + args.exp_name + "/actGradExtractionPlot/actGradExtractionPlotAG",
                        exist_ok=True)

            total_areas = 9
            args.num_classes = 9
            test_dataset = Sinthcity(partition='test', num_points=args.num_points, test_area=args.test_area)
            test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            preds = np.genfromtxt('checkpoints/' + args.exp_name + "/prediction.txt", delimiter=' ').astype("int64")

            # plot gradcam
            # for cls in range(0, 14):
            #idx = np.array(configSettings.BATCH_IDX)
            for cls in range(0, configSettings.OUTPUT_CHANNELS):
                print("Start Class " + str(cls))
                i = 0
                # test_loader = deepcopy(zip(objs, labs))
                # for data, gt in test_loader:

                for data, seg, max in test_loader:
                    #if (i in idx):
                        select_idx = []
                       
                        j = 0
                        #print(data.shape)
                        for points,cls2 in itertools.zip_longest(data,seg):
                            #print(data[0][j])
                            for points2,cls3 in itertools.zip_longest(points,cls2):
                              #print(points2.shape)
                              if cls3 in [8]: 
                                  select_idx.append(j)
                              j+=1
                        if len(select_idx) in [0,1,2]: 
                          i+=1
                          continue 
                        indices = torch.tensor(select_idx,dtype=torch.int32)
                        data = torch.index_select(data, 1, indices)
                        
                        #print(data.shape)
                        #print(data_final.shape)
                        #data = torch.FloatTensor(1, len(data_final), 9) 
                        #data[0] = data_final
                        #seg.remove(1)
                        #seg.remove(2)
                        #seg.remove(3)
                        #seg.remove(4)
                        #seg.remove(6)
                        #seg.remove(7)
                        #data = data_final
                        #print(data.shape)
                        
                        a = np.load('checkpoints/' + args.exp_name + "/actGradExtraction/act_conv7_{}.npy".format(i))
                        k=0
                        app = np.arange(len(select_idx)*512).reshape(1,512,len(select_idx)).astype('float')
                        for el in a[0]:
                          app[0][k] = np.take(el, select_idx)
                          k+=1
                        a = app

                        g = np.load(
                            'checkpoints/' + args.exp_name + "/actGradExtraction/grad_conv7_{}_tg{}.npy".format(i, cls))
                        app = np.arange(len(select_idx)*512).reshape(1,512,len(select_idx)).astype('float')
                        k=0
                        for el in g[0]:
                          app[0][k] = np.take(el, select_idx)
                          k+=1
                        g = app

                        ag = a * g
                        agM = np.median(ag, axis=1)

                        # aM= np.median(a,axis=1)
                        # gM = np.median(g, axis=1)
                        # agM = aM * gM

                        # gM = np.median(g, axis=1)
                        # ag = a * gM[:,np.newaxis,:]
                        # agM = np.median(ag, axis=1)

                        # aM = np.median(a, axis=1)
                        # ag = g * aM[:,np.newaxis,:]
                        # agM = np.median(ag, axis=1)

                        var = agM[0]

                        min_v = np.min(var)
                        max_v = np.max(var)
                        # gt = labs[i]
                        # pred = preds[i]

                        data[:, [1, 2]] = data[:, [2, 1]]

                        # varst = (var - min_v) / (max_v - min_v)  # +0.000001)

                        # simmetrizzazione
                        abs_max_v = np.maximum(abs(min_v), abs(max_v))
                        min_v = -abs_max_v
                        max_v = abs_max_v
                        varst = (var - min_v) / (max_v - min_v)  # +0.000001)
                        #print(varst)
                        ply = data  # numpy.stack((data, axis=-1)
                        pcd = o3d.geometry.PointCloud()

                        cmap = plt.cm.get_cmap("jet")
                        varst = cmap(varst)[:, :3]
                        #print(varst.shape)
                        #print(ply[0][:, :3].shape)
                        pcd.points = o3d.utility.Vector3dVector(ply[0][:, :3])
                        pcd.colors = o3d.utility.Vector3dVector(varst)
                        
                        if i < 10:
                            o3d.io.write_point_cloud('checkpoints/' + args.exp_name + "/actGradExtractionPlot/actGradExtractionPlotAG/ag_median_00{}_tg{}.ply".format(i, cls), pcd)
                        elif i < 100:
                            o3d.io.write_point_cloud('checkpoints/' + args.exp_name + "/actGradExtractionPlot/actGradExtractionPlotAG/ag_median_0{}_tg{}.ply".format(i, cls), pcd)
                        else:
                            o3d.io.write_point_cloud('checkpoints/' + args.exp_name + "/actGradExtractionPlot/actGradExtractionPlotAG/ag_median_{}_tg{}.ply".format(i, cls), pcd)
                        
                        print(i)
                        i += 1
                    #i += 1
            # #plot only activation
            # i=0
            # for data, gt in test_loader:
            #     a = np.load("results/actGradExtraction/act_conv5_{}.npy".format(idx[i]))

            #     aM = np.median(a, axis=1)
            #     var = aM[0]

            #     min_v = np.min(var)
            #     max_v = np.max(var)
            #     gt = labs[i]
            #     pred = preds[i]

            #     data[:, [1, 2]] = data[:, [2, 1]]

            #     # varst = (var - min_v) / (max_v - min_v)  # +0.000001)

            #     # simmetrizzazione
            #     abs_max_v= max(abs(min_v),abs(max_v))
            #     min_v= -abs_max_v
            #     max_v = abs_max_v
            #     varst = (var - min_v) / (max_v - min_v)  # +0.000001)

            #     ply = data  # numpy.stack((data, axis=-1)
            #     pcd = o3d.geometry.PointCloud()

            #     cmap = plt.cm.get_cmap("jet")
            #     varst = cmap(varst)[:, :3]

            #     pcd.points = o3d.utility.Vector3dVector(ply)
            #     pcd.colors = o3d.utility.Vector3dVector(varst)

            #     o3d.io.write_point_cloud("results/activationPlot/a_median_{}_gt{}_p{}.ply".format(idx[i], gt, pred), pcd)

            #     print(i)
            #     i += 1


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


explain_gradcam_semseg(args)