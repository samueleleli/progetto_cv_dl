import configSettings
import numpy as np
from numpy import save
# save to npy file

objstot = np.load(configSettings.DATASET_OBJS)
objs = objstot[:,0:6]
save('./data/synthcity/objs_Area_3.npy', objs)

objstot = np.load(configSettings.DATASET_OBJS)
objstot[:,6].tofile("./data/synthcity/labs_Area_3.txt","\n")


