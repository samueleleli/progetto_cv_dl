# dataset area synthcity (file gen_objs_labs_txt.py)
DATASET_TO_SPLIT = "data/synthcity/Area_3_Area_3.npy"
OUTPUT_OBJS = "./data/synthcity/objs_Area_3.npy"
OUTPUT_LABS = "./data/synthcity/labs_Area_3.txt"


DATASET_TO_PREDICT = "synthcity" # modelnet15 o modelnet40
DATASET_OBJS = "data/synthcity/objs_Area_3.npy"
DATASET_LABS = "data/synthcity/labs_Area_3.txt" 
OUTPUT_CHANNELS = 9

TRAINING_DATASET = "synthcity" # scanObjectNN

MODEL_PATH = "models/model_3_acc.t7" # models/model.t7
MODEL = 'dgcnn_semseg'

TEST_DATASET = "synthcity"
EXP_DIR = "test_output"
TEST_AREA = '3'
TEST_BATCH_SIZE = 1 # DA VERIFICARE
EVAL = True
MODEL_ROOT = "models"
PARALLEL = True
NUM_POINTS = 4096
USE_SGD = True
LR = 0.001
MOMENTUM = 0.9
SCHEDULER = "cos"
SEED = 1
EXTRACT = False   
   
# DATASET = "scanObjectNN"    # modelnet40
# CLASS_MAP = ['bag','bin','box','cabinet','chair','desk','display','dor','shelf','table','bed','pillow','sink','sofa','toilet']
# OUTPUT_CHANNELS = len(CLASS_MAP)