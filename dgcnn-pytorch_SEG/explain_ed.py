import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import cm
from collections import Counter
import argparse
import glob

########################## INPUT #######################
'''
layers = [5]   #[1, 2, 3, 4, 5]   # lista in cui inserire i layer da utilizzare: da 1 a 5

dataset = "synthcity"   # arch | synthcity | s3dis
base_path = "checkpoints/synthcity_extract_3_acc/"

classe = 5
'''
########################################################

def unione_nuvole(base_path, layer_number):
    '''
    Unisco le nuvole prodotte da explain_pca.py; creo un'unica nuvola, per ogni layer, aggiungengo le colonne con le distanze calcolate
    per ogni classe.
    Quindi alla fine avrà la nuvola: x y z gt pred dist0 dist1 dist2 ... distn, chiamata total_distance_layer1.txt

    :param base_path: dove prendere le nuvole delle singole classi
    :return:
    '''
    print("UNIONE NUVOLE ED:")

    layer = "distance_layer{}_class".format(layer_number)
    print("-"+layer)
    classi = sorted(glob.glob(base_path+layer+"*"))
    distanze = []
    for c in classi:
        distanze.append([])

    #save_coords = True
    print(" - lettura distanze...")
    for c in classi:
        base = os.path.basename(c)
        print("   -"+base)
        classe = int(base[21:base.index(".")])
        with open(c,"r") as fr:
            for l in fr:
                x,y,z,gt,pred,dist = l.strip().split(" ")
                distanze[classe].append(dist)

    lunghezze = []
    for d in distanze:
        lunghezze.append(len(d))
    print(" - check distanze: {}".format(lunghezze))

    print(" - salvataggio...")
    with open(base_path+"total_"+layer[:15]+".txt", "w") as fw:
        with open(classi[0],"r") as fr:
            i=0
            for l in fr:
                x,y,z,gt,pred,dist = l.strip().split(" ")
                fw.write("{} {} {} {} {}".format(x,y,z,gt,pred))
                for j in range(len(distanze)):
                    fw.write(" {}".format(distanze[j][i]))
                i+=1
                fw.write("\n")

    print(" - eliminazione nuvole temporanee...")
    for c in classi:
        os.remove(c)


def explain(layer, dataset, base_path, classe, union_remove):
    feat_path = base_path + "feat_{}.txt".format(layer)
    gt_path = base_path + "prediction.txt"
    save_path = base_path + "explain_ed/"
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
        #dizio = {}
        print("Loading GT...")
        with open(gt_path, "r") as fr:
            for l in fr:
                x, y, z, r, g, b, gt, pred = l.strip().split()
                #ipred = int(pred)
                preds.append([x,y,z,int(gt),int(pred)])

                #if ipred not in dizio:
                #    dizio[ipred] = []
                #dizio[ipred].append(cont)

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
    print("COORDS+LABELS: {}".format(len(preds)))



    if classe==-1:
        classi = [i for i in range(len(CLASS_MAP))]
    else:
        classi = [classe]

    for classe in classi:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        path_punto = save_path + "point_class{}.txt".format(classe)

        if os.path.exists(path_punto):
            print("Carico il punto a caso della classe {}...".format(classe), current_time)
            with open(path_punto,"r") as fr:
                for l in fr:
                    #pos = int(l.strip())
                    pos, x, y, z, gt, pred = l.strip().split(" ")
                    print(" - punto dal file ({},{},{}) gt:{} pred:{}".format(x, y, z, gt, pred))
                    pos = int(pos)

            punto = preds[pos]
            print(" - punto ({},{},{}) nella pos. {}...".format(punto[0], punto[1], punto[2], pos))
        else:
            print("Prendo un punto a caso della classe {}...".format(classe), current_time)


            #npy = np.array(preds)
            #a = (np.where(npy[:,3] == str(classe)) and np.where(npy[:,4] == str(classe)))
            npy = np.array(preds)
            gt = npy[:,3]
            pred = npy[:,4]
            #a = (np.where(gt == str(classe)) and np.where(pred == str(classe)))
            gt_where = np.where(gt == str(classe))
            pred_where = np.where(pred == str(classe))
            a = np.intersect1d(gt_where, pred_where)
            #pos = np.random.choice(a[0])
            pos = np.random.choice(a)

            punto = preds[pos]
            with open(path_punto,"w") as fw:
                fw.write("{} {} {} {} {} {}\n".format(pos, punto[0], punto[1], punto[2], punto[3], punto[4]))   # pos x y z gt pred


            print(
                " - punto ({},{},{}) nella pos. {} --> labels: {} - {}".format(punto[0], punto[1], punto[2], pos,
                                                                               punto[3],
                                                                               punto[4]))


        feat_pos = feats[pos]

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

        distanze=[]
        for p in feats:
            d = np.linalg.norm(feat_pos - p)  # x y z gt features
            distanze.append(d)


        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

        distance_np = np.array(distanze)
        minimo = np.min(distance_np)
        massimo = np.max(distance_np)
        dev_standard = np.std(distance_np)

        print("MIN: {} - MAX: {} - DEV: {}".format(minimo, massimo, dev_standard))


        hist_points, hist_limits = np.histogram(distanze, bins=20)
        print("- Histogram:\n - points {} \n - limits {}\n".format(hist_points, hist_limits))

        plt.clf()
        plt.hist(distanze, bins=20)
        out_plot = save_path + "histogram_layer{}_class{}.png".format(layer, classe)
        plt.title("{} - da {:.3f} a {:.3f}".format(layer, minimo, massimo))
        plt.savefig(out_plot)
        plt.close()


        print("salvo file delle distanze")
        out_file = save_path + "distance_layer{}_class{}.txt".format(layer, classe)
        with open(out_file, "w") as fw:
            for i in range(len(preds)):
                p = preds[i]
                dist = distance_np[i]
                fw.write(
                    "{} {} {} {} {} {}\n".format(p[0], p[1], p[2], int(p[3]), int(p[4]), dist))


    if union_remove:
        unione_nuvole(save_path, layer)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Done!!",current_time)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud XAI')
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name',
                        choices=["s3dis", "arch", "arch9l", "synthcity"])
    parser.add_argument('--layers', type=str, default='1', help='layers 1,2,3,4,5')     #default='1,2,3,4,5'
    parser.add_argument('--base_path', type=str, default='E:/MassimoMartini/Explainability/RISULTATI/xai_s3dis_extract_6/', help='Path for the inputs and outputs')
    #parser.add_argument('--use_jet', type=bool, default=False, help='Choice to use jet colormap')
    parser.add_argument('--type_class', type=int, default=-1, help='Specify the class of the random point that will be randomly choosen; -1 for all classes')
    #parser.add_argument('--label', type=str, default='gt', help='Type of label (gt,pred)', choices=["gt", "pred", "both"])
    parser.add_argument('--union_remove', type=bool, default=False, help='union_remove')

    args = parser.parse_args()

    if not os.path.exists(args.base_path):
        print("Errore: {} NON ESISTE".format(args.base_path))
    else:

        layers = args.layers.split(",")
        layers = [int(l) for l in layers]


        for layer in layers:
            explain(layer, args.dataset, args.base_path, args.type_class, args.union_remove)