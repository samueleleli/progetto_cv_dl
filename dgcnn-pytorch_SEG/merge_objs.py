import glob
import numpy as np

# cd /Users/Lisa/Documents/Visual_Studio_Code/progetto_cv_dl/dgcnn-pytorch_SEG/data/sinthcity-npy/sinthcity

objs = []
sorted_objs = sorted(glob.glob("*.npy"))
for area in sorted_objs:
    obj = np.load(area)
    objs = np.append(objs, obj)
    print(area)

# np.save('merge.npy',objs)


print(obj.size)

# np.save('merge.npy',objs)




# -------------
# a = np.load('a.npy')
# b = np.load('b.npy')
# c = []

# c = np.append(a,b)

# np.save('merge.npy',c)
