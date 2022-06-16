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


# DATASET = "scanObjectNN"    # modelnet40
# CLASS_MAP = ['bag','bin','box','cabinet','chair','desk','display','dor','shelf','table','bed','pillow','sink','sofa','toilet']
# OUTPUT_CHANNELS = len(CLASS_MAP)