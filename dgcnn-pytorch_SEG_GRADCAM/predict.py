import os
import argparse
from cv2 import exp
import visual_3d
import numpy
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from data import S3DIS, ArCH, Sinthcity, ModelNet40, ShapeNetPart
from model import DGCNN_semseg, DGCNN_cls
from torch.utils.data import DataLoader
from gradcam_exp import gradcam

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gradcam_exp.attgrad import ActivationsAndGradients
from sklearn.metrics import classification_report, confusion_matrix

import paramSettings
import configSettings
from util import IOStream

os.makedirs("results/classification",exist_ok=True)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def extract_cls(args):
    if True:
        if True:
            objs = np.load(configSettings.DATASET_OBJS)
            labs = np.genfromtxt(configSettings.DATASET_LABS, delimiter=' ').astype("int64")

            objs = objs[np.newaxis,:,:]
            labs = labs[np.newaxis,:]

            test_loader= zip(objs,labs) ##

            #####

            if not args.no_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Try to load models
            if args.model == 'dgcnn_cls':
                model = DGCNN_cls(args).to(device)
            elif args.model == 'dgcnn_semseg':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")

            model = nn.DataParallel(model)

            print(os.path.join(args.model_path))
            if args.model_path == "":
                print(os.path.join(args.model_root, 'model_%s.t7' % test_area))
                model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            else:
                if not args.no_cuda:
                    model.load_state_dict(torch.load(os.path.join(args.model_path)))
                else:
                    model.load_state_dict(torch.load(os.path.join(args.model_path),map_location = torch.device('cpu')))

            #model = model.train()
            model = model.eval()
            target_layer = model.module.conv5 #linear3 #linear2 #linear1 #conv5
            if not args.no_cuda:
                model = model.cuda()

            activations_and_grads = ActivationsAndGradients(model, target_layer, None)

            print(model)
            print("Model defined...")
            i = 0
            ACTIVATIONS = []
            gts = []
            predicts = []
            for data, gt in test_loader:

                data= torch.tensor([data])

                if not args.no_cuda:
                    data = data.cuda()
                    
                print(data)
                data = data.permute(0,2,1).to(device) # permute(0, 2, 1) classificazione  // prima noi (1,0) per fare trasposta
                print(data)
                
                output = activations_and_grads(data)

                am, idx = torch.max(output, 1)
                output = idx
                #output = torch.argmax(output.squeeze())

                #model.zero_grad()
                #loss = torch.mean(get_loss(output, target_category))
                #loss.backward(retain_graph=True)

                activations = activations_and_grads.activations[-1].cpu().data.numpy()
                #grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

                #np.save("classification\\act_conv5_{}.npy".format(i), activations)
                ACTIVATIONS.append(activations)
                #gts.append(gt[0].cpu().data.numpy().astype('int64'))
                gts.append(gt)
                predicts.append(output.cpu().data.numpy().astype('int64'))

                print("sample " + str(i) + " DONE")
                i+=1

                # if i==3:
                #   break

            ACTIVATIONS = np.concatenate(ACTIVATIONS)
            gts= np.array(gts)
            predicts= np.concatenate(predicts)

            numpy.savetxt('results/classification/gts.txt', gts, delimiter=" ",fmt='%d')
            numpy.savetxt('results/classification/predicts.txt', predicts, delimiter=" ",fmt='%d')
            numpy.savetxt('results/classification/ACT_linear2.txt', ACTIVATIONS, delimiter=" ",fmt='%.6f')


            print(classification_report(gts, predicts, target_names=CLASS_MAP))
            cm=confusion_matrix(gts, predicts)
            print(cm)
            plot_confusion_matrix(cm,CLASS_MAP,title='Confusion matrix',normalize=False,save_path="cm.png")

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

def not_unique(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def test_semseg(args, io):
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
            if args.model == 'dgcnn_semseg':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")

            if args.parallel:
                model = nn.DataParallel(model)

            if args.model_path=="":
                model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            else:
                model.load_state_dict(torch.load(os.path.join(args.model_path), map_location=torch.device("cuda" if args.cuda else "cpu")))

            model = model.eval()
            target_layer = model.module.conv9 
            activations_and_grads = ActivationsAndGradients(model, target_layer, None)
            ACTIVATIONS = []
            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            cont=0
            print("Test: {} batches".format(len(test_loader)))
            print("Num_points: " + str(configSettings.NUM_POINTS) + "\n" + "Batch size: " + str(configSettings.TEST_BATCH_SIZE) + "\n")
            with open('checkpoints/' + args.exp_name+ "/prediction.txt", "w") as fw:
                for data_or, seg, max in test_loader:
                    if cont % 100 == 0: print(cont)
                    cont += 1
                    data, seg = data_or.to(device), seg.to(device)
                    data = data.permute(0, 2, 1)

                    output = activations_and_grads(data)
                    output = output.mean(2)
                    am, idx = torch.max(output, 1) # torch.max prende valore massimo tensore
                    output = idx # idx indice (posizione) del valore massimo
                    activations = activations_and_grads.activations[-1].cpu().data.numpy()
                    activations = activations.mean(2)
                    ACTIVATIONS.append(activations)

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
            
            ACTIVATIONS = np.concatenate(ACTIVATIONS)
            numpy.savetxt('checkpoints/' + args.exp_name+ "/ACT_conv9.txt", ACTIVATIONS, delimiter=" ",fmt='%.6f')
            
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
            cm = metrics.confusion_matrix(test_true_cls,test_pred_cls)
            plot_confusion_matrix(cm, CLASS_MAP, title='Confusion matrix', normalize=False, save_path="checkpoints/"+configSettings.EXP_DIR+"/confusion_matrix/cm.png")
            classification_report = metrics.classification_report(test_true_cls, test_pred_cls, target_names=test_dataset.class_names, digits=3)
            io.cprint(str(classification_report))
            if configSettings.UNIQUE_RESULTS:
                with open('checkpoints/' + args.exp_name+ "/test_results.txt", "w") as res:
                    res.write("Test: {} batches".format(len(test_loader)) + "\n")
                    res.write("Num_points: " + str(configSettings.NUM_POINTS) + "\n" + "Batch size: " + str(configSettings.TEST_BATCH_SIZE) + "\n")
                    res.write(outstr + "\n")
                    res.write(classification_report)
            else:
                with open(not_unique('checkpoints/' + args.exp_name+ "/test_results.txt"), "w") as res:
                    res.write("Test: {} batches".format(len(test_loader)) + "\n")
                    res.write("Num_points: " + str(configSettings.NUM_POINTS) + "\n" + "Batch size: " + str(configSettings.TEST_BATCH_SIZE) + "\n")
                    res.write(outstr + "\n")
                    res.write(classification_report)
            print("Saving Plot...")
            visual_3d.save_plot_area_3d_pred_map_color(True)

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


if(configSettings.DATASET_TO_PREDICT == "modelnet15"):
    CLASS_MAP = ['bag','bin','box','cabinet','chair','desk','display','dor','shelf','table','bed','pillow','sink','sofa','toilet']
elif(configSettings.DATASET_TO_PREDICT == "modelnet40"):
    CLASS_MAP = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
elif(configSettings.DATASET_TO_PREDICT == "synthcity"):
    CLASS_MAP = ["building", "car", "natural-ground", "ground", "pole-like", "road", "street-furniture", "tree", "pavement"]
else: print("there is no CLASS_MAP defined")


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



os.makedirs("checkpoints/" + args.exp_name,exist_ok=True)
os.makedirs("checkpoints/"+args.exp_name+"/confusion_matrix",exist_ok=True)
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


if(args.model == 'dgcnn_cls'):
    extract_cls(args)
elif(args.model == 'dgcnn_semseg'):
    test_semseg(args,io)
