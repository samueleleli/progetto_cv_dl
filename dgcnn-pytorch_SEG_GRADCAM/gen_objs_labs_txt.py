import configSettings
import numpy as np
from numpy import save
# save to npy file

objstot = np.load(configSettings.DATASET_TO_SPLIT)
objs = objstot[:,0:6]
save(configSettings.OUTPUT_OBJS, objs)

# objstot[:,6].tofile("./data/synthcity/labs_Area_3.txt","\n")
labs = objstot[:,6].astype(int).tofile(configSettings.OUTPUT_LABS,"\n")
