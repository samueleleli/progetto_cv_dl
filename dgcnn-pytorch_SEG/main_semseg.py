#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_semseg.py
@Time: 2020/2/24 7:17 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DIS, ArCH, Sinthcity
from model import DGCNN_semseg, DGCNN_semseg_extract
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import shutil
from datetime import datetime

def backup(args):
    '''
    os.system('cp main_semseg.py checkpoints' + '/' + args.exp_name + '/' + 'main_semseg.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    '''

    #Ho aggiornato i metodi per il backup, per risolvere compatiilità con windows
    shutil.copy("main_semseg.py", os.path.join("checkpoints", args.exp_name, "main_semseg.py.backup"))
    shutil.copy("model.py", os.path.join("checkpoints", args.exp_name, "model.py.backup"))
    shutil.copy("util.py", os.path.join("checkpoints", args.exp_name, "util.py.backup"))
    shutil.copy("data.py", os.path.join("checkpoints", args.exp_name, "data.py.backup"))

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

    backup(args)



def calculate_sem_IoU(pred_np, seg_np, num_classes):
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all

def tempo(i):
    '''
    Funzione per stampare i timestamp, utilizzata per studiare i colli di bottiglia
    :param i:
    :return:
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(i,") Current Time =", current_time)


def train(args, io):
    if args.dataset == "S3DIS":
        args.num_classes=13
        train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area),
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == "ArCH":
        args.num_classes=10
        train_loader = DataLoader(ArCH(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ArCH(partition='test', num_points=args.num_points, test_area=args.test_area),
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == "ArCH9l":
        args.num_classes=9
        train_loader = DataLoader(ArCH(partition='train', num_points=args.num_points, test_area=args.test_area, tipo="9l"),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ArCH(partition='test', num_points=args.num_points, test_area=args.test_area, tipo="9l"),
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == "synthcity":
        args.num_classes=9
        train_loader = DataLoader(Sinthcity(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(Sinthcity(partition='test', num_points=args.num_points, test_area=args.test_area),
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    if args.parallel:
        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    best_test_iou = 0
    best_test_acc = 0
    best_epoch_iou = 0
    best_epoch_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("EPOCA {} di {}... - {}".format(epoch, args.epochs, current_time))
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        cont = 0
        print("Train: {} batches".format(len(train_loader)))

        for data, seg in train_loader:
            if cont%100==0: print(cont)
            cont+=1
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_classes), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        print(cont)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg, args.num_classes)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        cont=0
        print("Test: {} batches".format(len(test_loader)))
        for data, seg, max in test_loader:
            if cont%100==0: print(cont)
            cont+=1
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_classes), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        print(cont)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg, args.num_classes)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s_iou.t7' % (args.exp_name, args.test_area))
            print(" --> BEST IoU MODEL SAVED on epoch {}!".format(epoch))
            best_epoch_iou=epoch
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s_acc.t7' % (args.exp_name, args.test_area))
            print(" --> BEST Acc MODEL SAVED on epoch {}!".format(epoch))
            best_epoch_acc=epoch

    torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s_final.t7' % (args.exp_name, args.test_area))
    '''
    print(" --> FINAL MODEL SAVED on epoch {}!".format(epoch))
    print(" --> BEST IoU MODEL SAVED on epoch {}!".format(best_epoch_iou))
    print(" --> BEST Acc MODEL SAVED on epoch {}!".format(best_epoch_acc))
    '''
    outstr = " --> FINAL MODEL SAVED on epoch {}!".format(epoch)
    io.cprint(outstr)
    outstr = " --> BEST IoU MODEL SAVED on epoch {}!".format(best_epoch_iou)
    io.cprint(outstr)
    outstr = " --> BEST Acc MODEL SAVED on epoch {}!".format(best_epoch_acc)
    io.cprint(outstr)

def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    if args.dataset == "S3DIS":
        total_areas=6
        args.num_classes = 13
    elif args.dataset == "ArCH":
        total_areas=17
        args.num_classes = 10
    elif args.dataset == "ArCH9l":
        total_areas=17
        args.num_classes = 9
    elif args.dataset == "synthcity":
        total_areas=9
        args.num_classes = 9

    for test_area in range(1,total_areas+1):
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            if args.dataset == "S3DIS":
                test_dataset = S3DIS(partition='test', num_points=args.num_points, test_area=test_area)
                test_loader = DataLoader(test_dataset,
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "ArCH":
                test_dataset = ArCH(partition='test', num_points=args.num_points, test_area=test_area)
                test_loader = DataLoader(test_dataset,
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "ArCH9l":
                test_dataset = ArCH(partition='test', num_points=args.num_points, test_area=test_area, tipo="9l")
                test_loader = DataLoader(test_dataset,
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "synthcity":
                test_dataset = Sinthcity(partition='test', num_points=args.num_points, test_area=test_area)
                test_loader = DataLoader(test_dataset,
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)


            device = torch.device("cuda" if args.cuda else "cpu")

            #Try to load models
            if args.model == 'dgcnn':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")

            if args.parallel:
                model = nn.DataParallel(model)

            if args.model_path=="":
                model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            else:
                model.load_state_dict(torch.load(os.path.join(args.model_path)))

            model = model.eval()
            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            cont=0
            print("Test: {} batches".format(len(test_loader)))
            with open('checkpoints/' + args.exp_name+ "/prediction.txt", "w") as fw:
                for data_or, seg, max in test_loader:
                    if cont % 100 == 0: print(cont)
                    cont += 1
                    data, seg = data_or.to(device), seg.to(device)
                    data = data.permute(0, 2, 1)
                    batch_size = data.size()[0]
                    seg_pred = model(data)
                    #seg_pred, x1, x2, x3 = model(data)
                    #print(x1.detach().cpu().numpy().shape)  #(2, 64, 4096)  aggiungerei il permute 0,2,1 ...
                    #("wait...")
                    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                    pred = seg_pred.max(dim=2)[1]
                    seg_np = seg.cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    test_true_cls.append(seg_np.reshape(-1))
                    test_pred_cls.append(pred_np.reshape(-1))
                    test_true_seg.append(seg_np)
                    test_pred_seg.append(pred_np)

                    b, npoints, feats = data_or.shape
                    #preds = pred_np.reshape(-1)
                    #gts = seg_np.reshape(-1)
                    #print(max)
                    for bb in range(b):
                        for npp in range(npoints):
                            feat = data_or[bb,npp,:]
                            #print(feat)
                            #print(feat[6])
                            #print(max[0])
                            #print(max[0][bb])
                            px = feat[6] * max[0][bb]     #max[bb,npp,0]
                            py = feat[7] * max[1][bb]     #max[bb,npp,1]
                            pz = feat[8] * max[2][bb]     #max[bb,npp,2]
                            rgb = feat[3:6] * 255.0
                            pred = int(pred_np[bb,npp])
                            gt = int(seg_np[bb, npp])
                            fw.write("{} {} {} {} {} {} {} {}\n".format(px, py, pz, int(rgb[0]), int(rgb[1]), int(rgb[2]), gt, pred))
            print(cont)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg, args.num_classes)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)
            classification_report = metrics.classification_report(test_true_cls, test_pred_cls, target_names=test_dataset.class_names, digits=3)
            io.cprint(str(classification_report))


    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg, args.num_classes)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


def save_feats(feats_np, feats_files):
    for i in range(len(feats_files)):
        #if i<4: continue
        feat = feats_np[i].permute(0, 2, 1).detach().cpu().numpy()
        with open(feats_files[i], "a") as fw:
            b, npoints, nfeats = feat.shape
            for bb in range(b):
                for npp in range(npoints):
                    feats = feat[bb, npp]
                    str=""
                    for ff in feats:
                        str = str + "{:.6f} ".format(ff)
                    fw.write("{}\n".format(str[:-1]))

def extract(args, io):

    tempo(0)

    if args.dataset == "S3DIS":
        total_areas=6
        args.num_classes = 13
    elif args.dataset == "ArCH":
        total_areas=17
        args.num_classes = 10
    elif args.dataset == "ArCH9l":
        total_areas=17
        args.num_classes = 9
    elif args.dataset == "synthcity":
        total_areas=9
        args.num_classes = 9

    for test_area in range(1,total_areas+1):
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            if args.dataset == "S3DIS":
                test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "ArCH":
                test_loader = DataLoader(ArCH(partition='test', num_points=args.num_points, test_area=test_area),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "ArCH9l":
                test_loader = DataLoader(ArCH(partition='test', num_points=args.num_points, test_area=test_area, tipo="9l"),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "synthcity":
                test_loader = DataLoader(Sinthcity(partition='test', num_points=args.num_points, test_area=test_area),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)


            device = torch.device("cuda" if args.cuda else "cpu")

            #Try to load models
            if args.model == 'dgcnn':
                model = DGCNN_semseg_extract(args).to(device)
            else:
                raise Exception("Not implemented")

            if args.parallel:
                model = nn.DataParallel(model)

            if args.model_path=="":
                model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            else:
                model.load_state_dict(torch.load(os.path.join(args.model_path)))

            model = model.eval()

            feats_files=[]
            for i in range(1,6):
                nome = "checkpoints/{}/feat_{}.txt".format(args.exp_name, i)
                file = open(nome, "w")
                file.close()
                feats_files.append(nome)
            cont=0
            print("EXTRACT: {} batches".format(len(test_loader)))
            with open('checkpoints/' + args.exp_name+ "/prediction.txt", "w") as fw:
                for data_or, seg, max in test_loader:
                    if cont % 10 == 0: print(cont)
                    if cont % 100 == 0: tempo(cont)
                    cont += 1
                    data, seg = data_or.to(device), seg.to(device)
                    data = data.permute(0, 2, 1)
                    batch_size = data.size()[0]
                    seg_pred, feats = model(data)

                    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                    pred = seg_pred.max(dim=2)[1]
                    seg_np = seg.cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    #feats_np = feats.detach().cpu().numpy()
                    save_feats(feats, feats_files)

                    b, npoints, feats = data_or.shape

                    for bb in range(b):
                        for npp in range(npoints):
                            feat = data_or[bb,npp,:]

                            px = feat[6] * max[0][bb]     #max[bb,npp,0]
                            py = feat[7] * max[1][bb]     #max[bb,npp,1]
                            pz = feat[8] * max[2][bb]     #max[bb,npp,2]
                            rgb = feat[3:6] * 255.0
                            pred = int(pred_np[bb,npp])
                            gt = int(seg_np[bb, npp])
                            fw.write("{} {} {} {} {} {} {} {}\n".format(px, py, pz, int(rgb[0]), int(rgb[1]), int(rgb[2]), gt, pred))
            tempo(cont)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS',"ArCH","ArCH9l","synthcity"])
    parser.add_argument('--test_area', type=str, default=None, metavar='N' )
                        #choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--extract', type=bool, default=False,
                        help='feature extraction')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--parallel', type=bool, default=False,
                        help='Use Multiple GPUs')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if args.eval:
        test(args, io)
    elif args.extract:
        extract(args, io)
    else:
        train(args, io)
