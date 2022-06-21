# dataset area synthcity (file gen_objs_labs_txt.py)
DATASET_TO_SPLIT = "data/synthcity/Area_3_Area_3.npy"
OUTPUT_OBJS = "./data/synthcity/objs_Area_3.npy"
OUTPUT_LABS = "./data/synthcity/labs_Area_3.txt"
# -----------------------------------------------------
DATASET_TO_PREDICT = "synthcity" # modelnet15 o modelnet40
DATASET_OBJS = "data/synthcity/objs_Area_3.npy" # non più utilizzato per il test preso da main_semseg
DATASET_LABS = "data/synthcity/labs_Area_3.txt" # non più utilizzato per il test preso da main_semseg
# -----------------------------------------------------

# variabili per TEST segmentazione
MODEL_PATH = "models/model_3_acc.t7" # models/model.t7  - modello allenato (su aree 1,24,5,6,7,8,9 di synthcity) all'epoca con maggior accuratezza 
MODEL = 'dgcnn_semseg' # architettura utilizzata: dgcnn per semantic segmentation
OUTPUT_CHANNELS = 9 # synthcity ha 9 classi
EXP_DIR = "test_output" # directory dove salvare output
TEST_DATASET = "synthcity" # dataset da predire (facciamo predizione su area 3)
TEST_AREA = '3'
TEST_BATCH_SIZE = 1 # provare anche 2 / 4 / 16 / 32 / 64 / 128
MODEL_ROOT = "models"
PARALLEL = True
NUM_POINTS = 4096 # cosa indica? l'intorno da considerare per convoluzione?
SEED = 1 # serve per pytorch CUDA
EXTRACT = False # serve nel main sem_seg per far partire il codice extract
EVAL = True # serve nel main sem_seg per far partire il codice di test

# variabili per TRAINING segmentazione (che noi non facciamo)
TRAINING_DATASET = "synthcity" # scanObjectNN
LR = 0.001
SCHEDULER = "cos"
USE_SGD = True # Stochastic gradient descent
MOMENTUM = 0.9

