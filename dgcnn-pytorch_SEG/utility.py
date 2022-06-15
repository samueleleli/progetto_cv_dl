import glob
import numpy as np
from collections import Counter
import os
from data import S3DIS, ArCH, Sinthcity
from torch.utils.data import DataLoader

def check_npy(scenes_path):
    scenes_list = glob.glob(scenes_path + "*.npy")
    l = len(scenes_list)
    print("\nTrovate {} scene:".format(l))
    if l > 0:
        for s in scenes_list:
            print("\n"+os.path.basename(s) + ":")
            scene = np.load(s)
            shape = scene.shape
            print(" - shape = {}".format(shape))
            #print(" - first line = {}".format(scene[0]))
            c = Counter(scene[:,6])
            print(" - Counter:{}".format(c))
            if 9.0 in c:
                print(" - c'è l'ultima classe")

scenes_path = "E:/MassimoMartini/pycharm/dgcnn.pytorch/data/arch9l-npy/arch9l/"

#check_npy(scenes_path)


def npy2txt(npy_path,out_path):
    npy=np.load(npy_path)
    print(npy_path,npy.shape)
    input()
    with open(out_path,"w") as f:
        for l in range(npy.shape[0]):
            if l%100000==0: print(l)
            s=""
            for c in range(npy.shape[1]):
                s=s+str(npy[l,c])+" "
            f.write(s[:-1]+"\n")

    print("Done!")

#npy2txt("E:/MassimoMartini/pycharm/dgcnn.pytorch/data/arch9l-npy/arch9l/Area_1_Area_1.npy","Area_1_Area_1.txt")


def test_h5_data(test_area, out_path, tipo="all"):
    test_loader = DataLoader(ArCH(partition='test', num_points=4096, test_area=test_area, tipo=tipo),
                             batch_size=2, shuffle=False, drop_last=False)

    print("EXTRACT: {} batches".format(len(test_loader)))
    cont=0
    labels = []
    with open(out_path, "w") as fw:
        for data_or, seg, max in test_loader:
            if cont % 10 == 0: print(cont)
            cont += 1

            b, npoints, feats = data_or.shape

            for bb in range(b):
                for npp in range(npoints):
                    feat = data_or[bb, npp, :]

                    px = feat[6] * max[0][bb]  # max[bb,npp,0]
                    py = feat[7] * max[1][bb]  # max[bb,npp,1]
                    pz = feat[8] * max[2][bb]  # max[bb,npp,2]
                    rgb = feat[3:6] * 255.0
                    gt = int(seg[bb, npp])
                    labels.append(gt)
                    fw.write(
                        "{} {} {} {} {} {} {}\n".format(px, py, pz, int(rgb[0]), int(rgb[1]), int(rgb[2]), gt))
    print(cont)
    print(Counter(labels))

test_h5_data("17", "area17-9l.txt","9l")
test_h5_data("17", "area17-all.txt", "all")