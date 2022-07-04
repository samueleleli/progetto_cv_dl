import os
import argparse
import paramSettings
import numpy
import numpy as np
import torch
import torch.nn as nn
from data import S3DIS, ArCH, Sinthcity, ModelNet40, ShapeNetPart
from model import DGCNN_semseg, DGCNN_cls
from torch.utils.data import DataLoader
from gradcam_exp import gradcam

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

import configSettings

# ******* plot grad *************
def plot_extract_semseg(args):
    os.makedirs('checkpoints/' + args.exp_name+ "/actGradExtraction",exist_ok=True)
    os.makedirs('checkpoints/' + args.exp_name+ "/actGradExtractionPlot",exist_ok=True)
    os.makedirs('checkpoints/' + args.exp_name+ "/actGradExtractionPlot/actGradExtractionPlotG",exist_ok=True)
    os.makedirs('checkpoints/' + args.exp_name+ "/actGradExtractionPlot/actGradExtractionPlotAG",exist_ok=True)
    os.makedirs('checkpoints/' + args.exp_name+ "/actGradExtractionPlot/actGradExtractionPlotA",exist_ok=True)
    total_areas = 9
    args.num_classes = 9
    test_dataset = Sinthcity(partition='test', num_points=args.num_points, test_area=args.test_area)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    ###
    if True:
        if True:
            preds = np.genfromtxt('checkpoints/' + args.exp_name+ "/prediction.txt", delimiter=' ').astype("int64")
            
            for cls in range(0, configSettings.OUTPUT_CHANNELS):
                i=0
                test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
                
                # for data, gt in test_loader:
                for data, seg, max in test_loader:

                    #g = np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\act_conv5_{}.npy".format(idx[i]))
                    g=np.load('checkpoints/' + args.exp_name+ "/actGradExtraction/grad_conv5_{}_tg{}.npy".format(i, cls))

                    # histogram of features across point
                    # histogram of point across features
                    H =[]
                    B= []
                    GM=[]
                    Gm=[]
                    M=[]
                    m=[]
                    for j in range(g.shape[2]):
                        h, b = np.histogram(g[0,:,j], bins=100)
                        H.append(h)
                        B.append(b)
                        GM.append(np.median(g[0,:,j]))
                        Gm.append(np.mean(g[0,:,j]))
                        M.append(np.max(g[0,:,j]))
                        m.append(np.min(g[0,:,j]))

                    # m= np.min(g, axis=1)
                    # M= np.max(g, axis=1)
                    # Gm= np.mean(g, axis=1)
                    # GM= np.median(g, axis=1)

                    plt.figure()
                    plt.plot(m)
                    plt.plot(M)
                    plt.plot(Gm)
                    plt.plot(GM)

                    plt.figure()
                    h, b = np.histogram(m, bins=100)
                    plt.stem(b[1:], h)

                    plt.figure()
                    h, b = np.histogram(M, bins=100)
                    plt.stem(b[1:], h)

                    plt.figure()
                    h, b = np.histogram(Gm, bins=100)
                    plt.stem(b[1:], h)

                    plt.figure()
                    h, b = np.histogram(GM, bins=100)
                    plt.stem(b[1:], h)

                    gM= np.median(g,axis=1)
                    #agM = np.mean(ag, axis=1)
                    var = gM[0]

                    min_v = np.min(var)
                    max_v = np.max(var)
                    gt = seg[0] # prende le 4096 etichette associate ai punti del batch i
                    pred = preds[i*4096:i*4096+4096, 7]

                    data[:, [1, 2]] = data[:, [2, 1]]

                    #varst = (var - min_v) / (max_v - min_v)  # +0.000001)

                    # simmetrizzazione
                    abs_max_v = np.maximum(abs(min_v), abs(max_v))
                    min_v = -abs_max_v
                    max_v = abs_max_v
                    varst = (var - min_v) / (max_v - min_v)  # +0.000001)

                    ply = data  # numpy.stack((data, axis=-1)
                    pcd = o3d.geometry.PointCloud()

                    cmap = plt.cm.get_cmap("jet")
                    varst = cmap(varst)[:, :3]

                    # pcd.points = o3d.utility.Vector3dVector(ply)
                    pcd.points = o3d.utility.Vector3dVector(ply[0][:,:3]) # ply tensore (0,4096,9), ply[0] accedo a matrice 4096x9 (9 n° feature?), prendiamo le prime tre colonne (coordinate xyz per i 4096 punti del batch)
                    pcd.colors = o3d.utility.Vector3dVector(varst)

                    o3d.io.write_point_cloud('checkpoints/' + args.exp_name + "/actGradExtractionPlot/actGradExtractionPlotG/g_MED6_{}_tg{}.ply".format(i, cls), pcd)

                    print(i)
                    i += 1
            
            '''
            # plot gradcam
            for cls in range(0, 40):
                i=0
                test_loader = deepcopy(zip(objs, labs))
                for data, gt in test_loader:
            
                    a=np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\act_conv5_{}.npy".format(idx[i]))
                    g=np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\grad_conv5_{}_tg{}.npy".format(idx[i], cls))
            
                    ag= a*g
                    agM= np.median(ag,axis=1)
            
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
                    gt = labs[i]
                    pred = preds[i]
            
                    data[:, [1, 2]] = data[:, [2, 1]]
            
                    # varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            
                    # simmetrizzazione
                    abs_max_v = max(abs(min_v), abs(max_v))
                    min_v = -abs_max_v
                    max_v = abs_max_v
                    varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            
                    ply = data  # numpy.stack((data, axis=-1)
                    pcd = o3d.geometry.PointCloud()
            
                    cmap = plt.cm.get_cmap("jet")
                    varst = cmap(varst)[:, :3]
            
                    pcd.points = o3d.utility.Vector3dVector(ply)
                    pcd.colors = o3d.utility.Vector3dVector(varst)
            
                    o3d.io.write_point_cloud(directory + "ag_mean6_{}_tg{}_gt{}_p{}.ply".format(idx[i], cls, gt, pred), pcd)
            
                    print(i)
                    i += 1

            # plot only activation
            i=0
            for data, gt in test_loader:
                a = np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\act_conv5_{}.npy".format(idx[i]))
            
                aM = np.min(a, axis=1)
                var = aM[0]
            
                min_v = np.min(var)
                max_v = np.max(var)
                gt = labs[i]
                pred = preds[i]
            
                data[:, [1, 2]] = data[:, [2, 1]]
            
                # varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            
                # simmetrizzazione
                abs_max_v= max(abs(min_v),abs(max_v))
                min_v= -abs_max_v
                max_v = abs_max_v
                varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            
            
                ply = data  # numpy.stack((data, axis=-1)
                pcd = o3d.geometry.PointCloud()
            
                cmap = plt.cm.get_cmap("jet")
                varst = cmap(varst)[:, :3]
            
                pcd.points = o3d.utility.Vector3dVector(ply)
                pcd.colors = o3d.utility.Vector3dVector(varst)
            
                o3d.io.write_point_cloud(directory + "a_min_{}_gt{}_p{}.ply".format(idx[i], gt, pred), pcd)
            
                print(i)
                i += 1
            '''

class args(object):
    model_path= configSettings.MODEL_PATH # "models/model.cls.1024.t7" # 
    model= configSettings.MODEL
    k = 20 # non utilizzato in segmentazione
    emb_dims= 1024 # non utilizzato in segmentazione
    dropout= 0 #0.5   # non utilizzato in segmentazione
    # num_classes = configSettings.OUTPUT_CHANNELS --- viene settato nel codice del test
    no_cuda= paramSettings.NO_CUDA
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

plot_extract_semseg(args)