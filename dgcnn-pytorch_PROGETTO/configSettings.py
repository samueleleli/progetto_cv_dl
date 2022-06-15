DATASET_TO_PREDICT = "modelnet15" # modelnet15 o modelnet40
DATASET_OBJS = "data/modelnet15/objs.npy"
DATASET_LABS = "data/modelnet15/GT.txt" 
OUTPUT_CHANNELS = 15

TRAINING_DATASET = "scanObjectNN"

MODEL_PATH = "models/model.t7"
MODEL = 'dgcnn_cls'


# DATASET = "scanObjectNN"    # modelnet40
# CLASS_MAP = ['bag','bin','box','cabinet','chair','desk','display','dor','shelf','table','bed','pillow','sink','sofa','toilet']
# OUTPUT_CHANNELS = len(CLASS_MAP)