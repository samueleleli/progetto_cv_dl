import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

esa = []

data = np.load('/data/sinthcity-npy/sinthcity/Area_3_Area_3.npy')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

#f44336 Rosso
#8fce00 Verde
#2986cc Blu
#fff400 Giallo
#ff00b7 Fucsia
#ff8f00 Arancione
#744700 Marrone
#6a329f Viola
#00ffe5 Celestino


for color in data:
  '''
  if color[6] == 0:
    esa.append("#f44336")
  elif color[6] == 1:
    esa.append("#8fce00")
  elif color[6] == 2:
    esa.append("#2986cc")
  elif color[6] == 3:
    esa.append("#fff400")
  elif color[6] == 4:
    esa.append("#ff00b7")
  elif color[6] == 5:
    esa.append("#ff8f00")
  elif color[6] == 6:
    esa.append("#744700")
  elif color[6] == 7:
    esa.append("#6a329f")
  elif color[6] == 8:
    esa.append("#00ffe5")
'''
  esa.append("#{0:02x}{1:02x}{2:02x}".format(int(color[3]), int(color[4]), int(color[5])))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c=esa)
plt.show()