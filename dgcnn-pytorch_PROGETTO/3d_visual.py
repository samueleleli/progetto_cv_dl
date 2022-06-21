import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches
'''
fig, ax = plt.subplots()
red_patch = mpatches.Patch(color='red', label='The red data')
ax.legend(handles=[red_patch])

plt.show()

'''

REAL_COLORS = False  # True

esa = []

data = np.load('data/sinthcity-npy/sinthcity/Area_3_Area_3.npy')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

if(REAL_COLORS):
  for color in data:
    esa.append("#{0:02x}{1:02x}{2:02x}".format(int(color[3]), int(color[4]), int(color[5])))
  ax.scatter(x, y, z, c=esa)
  # plt.show()
  plt.savefig('checkpoints/plot_output/real_colors.png')

else:
  for color in data:
    if color[6] == 0:
      esa.append("#f44336")  #f44336 Rosso
    elif color[6] == 1:
      esa.append("#8fce00")  #8fce00 Verde
    elif color[6] == 2:
      esa.append("#2986cc")  #2986cc Blu
    elif color[6] == 3:
      esa.append("#fff400")  #fff400 Giallo
    elif color[6] == 4:
      esa.append("#ff00b7")  #ff00b7 Fucsia
    elif color[6] == 5:
      esa.append("#ff8f00")  #ff8f00 Arancione
    elif color[6] == 6:
      esa.append("#744700")  #744700 Marrone
    elif color[6] == 7:
      esa.append("#6a329f")  #6a329f Viola
    elif color[6] == 8:
      esa.append("#00ffe5")  #00ffe5 Celestino
  
  ax.scatter(x, y, z, c=esa)
  # plt.show()
  plt.savefig('checkpoints/plot_output/colors_map.png')

  
  



