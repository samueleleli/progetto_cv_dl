import numpy as np
import pickle
import os

import random

from datetime import datetime
from collections import Counter
import argparse

########################## INPUT #######################
'''
layers = [1,2,3,5]   #[1, 2, 3, 4, 5]   # lista in cui inserire i layer da utilizzare: da 1 a 5

dataset = "arch"   # arch | synthcity | s3dis
base_path = "checkpoints/arch_extract_16_acc/"

'''

########################################################

def explain(layer, dataset, base_path, group, tipo_label):
    feat_path = base_path + "feat_{}.txt".format(layer)
    gt_path = base_path + "prediction.txt"
    save_path = base_path + "explain_l2norm/"
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path + tipo_label + "/"
    os.makedirs(save_path, exist_ok=True)

    map_feats_layer=[64,64,64,1024,256]
    N_FEATS = map_feats_layer[layer-1]

    if dataset == "synthcity":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0]
        }

        CLASS_MAP = ["building", "car", "natural-ground", "ground", "pole-like", "road", "street-furniture", "tree", "pavement"]

    elif dataset == "arch":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0],
            9: [100, 100, 255],
        }

        CLASS_MAP = ["arc", "column", "moulding", "floor", "door-window", "wall", "stairs", "vault", "roof", "other"]

    elif dataset == "arch9l":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0]
        }

        CLASS_MAP = ["arc", "column", "moulding", "floor", "door-window", "wall", "stairs", "vault", "roof"]

    elif dataset == "s3dis":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0],
            9: [100, 100, 255],
        }

        CLASS_MAP = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    else:
        input("Errore dataset: {}".format(dataset))

    print("\n\nLayer {} with {} features".format(layer, N_FEATS))

    # NON mi serve bilanciarlo, userò sempre tutti le features
    save_path = save_path + "all/"
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(save_path+"feats_{}.pickle".format(layer)):
        print("Carico i file esistenti...")
        with open(save_path+"feats_{}.pickle".format(layer), "rb") as f:
            feats, preds = pickle.load(f)
    else:

        preds = []
        cont = 0
        print("Loading GT...")
        with open(gt_path, "r") as fr:
            for l in fr:
                x, y, z, r, g, b, gt, pred = l.strip().split()
                preds.append([x,y,z,int(gt),int(pred)])

                cont += 1
                if cont % 100000 == 0:    print(cont)
        print(cont)


        feats = np.zeros((cont,N_FEATS),dtype="float32")

        i=0
        print("Loading FEATS...")
        with open(feat_path, "r") as fr:
            for l in fr:
                ff = l.strip().split()
                for j, fff in enumerate(ff):
                    feats[i,j] = float(fff)
                i += 1
                if i % 100000 == 0:    print(i)
        print(i)

        print("Salvo i file...")
        with open(save_path + "feats_{}.pickle".format(layer), "wb") as f:
            pickle.dump([feats, preds], f, protocol=4)


    print("FEATS: {}".format(feats.shape))
    print("LABELS: {}".format(len(preds)))
    if tipo_label == "gt":
        labels = [p[3] for p in preds]
        counter_labels = Counter(labels)
        print("GT Counter:", counter_labels)
    else:
        labels = [p[4] for p in preds]
        counter_labels = Counter(labels)
        print("PRED Counter:", counter_labels)

    print("\nGroup={}:".format(group))
    '''
    if group == "class":
        for l in sorted(counter_labels.keys()):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("\nLabel", l, ":", current_time)

            distanze = []
            for i in range(len(labels)):
                if labels[i] == l:
                    p = feats[i]
                    d = np.linalg.norm(p)  # x y z gt features
                    distanze.append(d)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            distance_np = np.array(distanze)

            print(" - salvo file delle distanze")
            out_file = save_path + "distance_layer{}_class{}.txt".format(layer, l)
            with open(out_file, "w") as fw:
                for i in range(len(preds)):
                    p = preds[i]
                    dist = distance_np[i]
                    fw.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], int(p[3]), int(p[4]), dist))

    else:
        distanze = []
        for p in feats:
            d = np.linalg.norm(p)  # x y z gt features
            distanze.append(d)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

        distance_np = np.array(distanze)

        print("salvo file delle distanze")
        out_file = save_path + "distance_all_layer{}.txt".format(layer)
        with open(out_file, "w") as fw:
            for i in range(len(preds)):
                p = preds[i]
                dist = distance_np[i]
                fw.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], int(p[3]), int(p[4]), dist))
    '''

    if group=="class":
        for l in sorted(counter_labels.keys()):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("\nLabel",l,":",current_time)

            out_file = save_path + "distance_layer{}_class{}.txt".format(layer, l)
            with open(out_file, "w") as fw:
                for i in range(len(labels)):
                    if labels[i]==l:
                        ff = feats[i]
                        dist = np.linalg.norm(ff)  # x y z gt features
                        p = preds[i]
                        fw.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], int(p[3]), int(p[4]), dist))

    else:
        out_file = save_path + "distance_all_layer{}.txt".format(layer)
        with open(out_file, "w") as fw:
            for i in range(len(labels)):
                ff = feats[i]
                dist = np.linalg.norm(ff)  # x y z gt features
                p = preds[i]
                fw.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], int(p[3]), int(p[4]), dist))


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Done!!",current_time)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud XAI')
    parser.add_argument('--dataset', type=str, default='arch', help='Dataset name',
                        choices=["s3dis", "arch", "arch9l", "synthcity"])
    parser.add_argument('--layers', type=str, default='1,2,3,4,5', help='layers 1,2,3,4,5')
    parser.add_argument('--base_path', type=str, default='', help='Path for the inputs and outputs')
    parser.add_argument('--group', type=str, default="class", help='Choice to use l2norm for every class or the entire point clouds (or both)',
                        choices = ["class", "all", "both"])
    parser.add_argument('--label', type=str, default='gt', help='Type of label (gt,pred)',
                        choices=["gt", "pred", "both"])

    args = parser.parse_args()

    if not os.path.exists(args.base_path):
        print("Errore: {} NON ESISTE".format(args.base_path))
    else:

        layers = args.layers.split(",")
        layers = [int(l) for l in layers]

        if args.label == "both":    labels = ["gt", "pred"]
        else:                       labels = [args.label]
        if args.group == "both":    groups = ["class", "all"]
        else:                       groups = [args.group]

        for tipo_label in labels:
            for group in groups:
                for layer in layers:
                    explain(layer, args.dataset, args.base_path, group, tipo_label)