import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import configSettings
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches
import open3d as o3d
import plotly.graph_objects as go


def save_plot_area_3d_real_color():
    os.makedirs("checkpoints/plot_output/", exist_ok=True)

    esa = []

    data = np.load('data/sinthcity-npy/sinthcity/Area_3_Area_3.npy')

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for color in data:
        esa.append("#{0:02x}{1:02x}{2:02x}".format(int(color[3]), int(color[4]), int(color[5])))

    ax.scatter(x, y, z, c=esa, s=0.1)
    # plt.show()
    plt.savefig("checkpoints/plot_output/real_colors.png")


def save_plot_area_3d_pred_map_color(predict):
    esa = []

    if predict:
        os.makedirs("checkpoints/" + configSettings.EXP_DIR + "/plot_output_pred/", exist_ok=True)
        data = np.loadtxt("checkpoints/" + configSettings.EXP_DIR + "/prediction.txt")
        output_file = "checkpoints/" + configSettings.EXP_DIR + "/plot_output_pred/" + "color_map_predict_label.png"
        index = 7
        building_color_0 = "#5c5c5c99"  # grigio trasparente #5c5c5ccc (80% opacità) # #5c5c5c80 (50% opacità) #5c5c5c99 60%
    else:
        os.makedirs("checkpoints/plot_output/", exist_ok=True)
        data = np.load('data/sinthcity-npy/sinthcity/Area_3_Area_3.npy')
        output_file = "checkpoints/plot_output/color_map_real_label.png"
        index = 6
        building_color_0 = "#5c5c5c33"  # grigio trasparente #5c5c5c33 (20% opacità)

    
    car_color_1 = "#fa0202"  # rosso
    natural_ground_color_2 = "#faa243"  # marroncino
    ground_color_3 = "#60849e"  # blu chiaro
    pole_like_color_4 = "#0f03fc"  # blu
    road_color_5 = "#2a2b2a"  # grigio
    street_forniture_color_6 = "#fc6603"  # arancione
    tree_color_7 = "#027013"  # verde
    pavement_color_8 = "#e4ed39"  # giallo

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for color in data:
        if color[index] == 0:
            esa.append(building_color_0)
        elif color[index] == 1:
            esa.append(car_color_1)
        elif color[index] == 2:
            esa.append(natural_ground_color_2)
        elif color[index] == 3:
            esa.append(ground_color_3)
        elif color[index] == 4:
            esa.append(pole_like_color_4)
        elif color[index] == 5:
            esa.append(road_color_5)
        elif color[index] == 6:
            esa.append(street_forniture_color_6)
        elif color[index] == 7:
            esa.append(tree_color_7)
        elif color[index] == 8:
            esa.append(pavement_color_8)

    ax.scatter(x, y, z, c=esa, s=0.1)
    building_patch = mpatches.Patch(color=building_color_0, label='Building')
    car_patch = mpatches.Patch(color=car_color_1, label='Car')
    natural_ground_patch = mpatches.Patch(color=natural_ground_color_2, label='Natural ground')
    ground_patch = mpatches.Patch(color=ground_color_3, label='Ground')
    pole_like_patch = mpatches.Patch(color=pole_like_color_4, label='Pole-like')
    road_patch = mpatches.Patch(color=road_color_5, label='Road')
    street_forniture_patch = mpatches.Patch(color=street_forniture_color_6, label='Street forniture')
    tree_patch = mpatches.Patch(color=tree_color_7, label='Tree')
    pavement_patch = mpatches.Patch(color=pavement_color_8, label='Pavement')
    ax.legend(handles=[building_patch, car_patch, natural_ground_patch, ground_patch, pole_like_patch, road_patch,
                       street_forniture_patch, tree_patch, pavement_patch], loc='upper left')

    plt.savefig(output_file)


def save_plot_area_3d_real_color():
    os.makedirs("checkpoints/plot_output/", exist_ok=True)

    esa = []

    data = np.load('data/sinthcity-npy/sinthcity/Area_3_Area_3.npy')

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for color in data:
        esa.append("#{0:02x}{1:02x}{2:02x}".format(int(color[3]), int(color[4]), int(color[5])))

    ax.scatter(x, y, z, c=esa, s=0.1)
    # plt.show()
    plt.savefig("checkpoints/plot_output/real_colors.png")


def plot_1_batch_a_time(predict, i_batch):
    esa = []

    if predict:
        os.makedirs("checkpoints/" + configSettings.EXP_DIR + "/plot_output_pred/", exist_ok=True)
        data = np.loadtxt("checkpoints/" + configSettings.EXP_DIR + "/prediction.txt")
        output_file = "checkpoints/" + configSettings.EXP_DIR + "/plot_output_pred/" + "color_map_predict_label.png"
        index = 7
        building_color_0 = "#5c5c5c99"  # grigio trasparente #5c5c5ccc (80% opacità) # #5c5c5c80 (50% opacità) #5c5c5c99 60%
    else:
        os.makedirs("checkpoints/plot_output/", exist_ok=True)
        data = np.load('data/sinthcity-npy/sinthcity/Area_3_Area_3.npy')
        output_file = "checkpoints/plot_output/color_map_real_label.png"
        index = 6
        building_color_0 = "#5c5c5c33"  # grigio trasparente #5c5c5c33 (20% opacità)

    
    car_color_1 = "#fa0202"  # rosso
    natural_ground_color_2 = "#faa243"  # marroncino
    ground_color_3 = "#60849e"  # blu chiaro
    pole_like_color_4 = "#0f03fc"  # blu
    road_color_5 = "#2a2b2a"  # grigio
    street_forniture_color_6 = "#fc6603"  # arancione
    tree_color_7 = "#027013"  # verde
    pavement_color_8 = "#e4ed39"  # giallo

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    i=i_batch
    k=0
    for color in data:
      if k>=i*4096 and k<(i*4096+4096):
        if color[index] == 0:
            esa.append(building_color_0)
        elif color[index] == 1:
            esa.append(car_color_1)
        elif color[index] == 2:
            esa.append(natural_ground_color_2)
        elif color[index] == 3:
            esa.append(ground_color_3)
        elif color[index] == 4:
            esa.append(pole_like_color_4)
        elif color[index] == 5:
            esa.append(road_color_5)
        elif color[index] == 6:
            esa.append(street_forniture_color_6)
        elif color[index] == 7:
            esa.append(tree_color_7)
        elif color[index] == 8:
            esa.append(pavement_color_8)
      else:
        esa.append("#FFFFFF00")
      k+=1
    ax.scatter(x, y, z, c=esa, s=0.1)
    building_patch = mpatches.Patch(color=building_color_0, label='Building')
    car_patch = mpatches.Patch(color=car_color_1, label='Car')
    natural_ground_patch = mpatches.Patch(color=natural_ground_color_2, label='Natural ground')
    ground_patch = mpatches.Patch(color=ground_color_3, label='Ground')
    pole_like_patch = mpatches.Patch(color=pole_like_color_4, label='Pole-like')
    road_patch = mpatches.Patch(color=road_color_5, label='Road')
    street_forniture_patch = mpatches.Patch(color=street_forniture_color_6, label='Street forniture')
    tree_patch = mpatches.Patch(color=tree_color_7, label='Tree')
    pavement_patch = mpatches.Patch(color=pavement_color_8, label='Pavement')
    ax.legend(handles=[building_patch, car_patch, natural_ground_patch, ground_patch, pole_like_patch, road_patch,
                       street_forniture_patch, tree_patch, pavement_patch], loc='upper left')

    plt.savefig(output_file)


def save_plot_GRADCAM(tg_class):
    esa = []
    x_coord = []
    y_coord = []
    z_coord = []

    os.makedirs("checkpoints/plot_output/", exist_ok=True)
    cloud_1 = o3d.io.read_point_cloud("checkpoints/"+configSettings.EXP_DIR+"/actGradExtractionPlot/actGradExtractionPlotAG/ag_median_000_tg"+str(tg_class)+".ply")
    list_ply = os.listdir("checkpoints/"+configSettings.EXP_DIR+"/actGradExtractionPlot/actGradExtractionPlotAG/")
    list_ply.remove("checkpoints")
    output_file = "checkpoints/plot_output/plot_GRADCAM_tg"+str(tg_class)+".png"


    colors = None
    if cloud_1.has_colors():
        colors = np.asarray(cloud_1.colors)
    elif cloud_1.has_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(cloud_ply.normals) * 0.5
    else:
        geometry.paint_uniform_color((1.0, 0.0, 0.0))
        colors = np.asarray(geometry.colors)
    for color in colors:
        esa.append("#{0:02x}{1:02x}{2:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255)))

    for ply_file in list_ply:
        cloud_ply = o3d.io.read_point_cloud("checkpoints/" + configSettings.EXP_DIR + "/actGradExtractionPlot/actGradExtractionPlotAG/" + ply_file)
        if ("tg"+str(tg_class)) in ply_file:
            if not "ag_median_000_" in ply_file:
                colors = None
                if cloud_ply.has_colors():
                    colors = np.asarray(cloud_ply.colors)
                elif cloud_ply.has_normals():
                    colors = (0.5, 0.5, 0.5) + np.asarray(cloud_ply.normals) * 0.5
                else:
                    geometry.paint_uniform_color((1.0, 0.0, 0.0))
                    colors = np.asarray(geometry.colors)

                #for color in colors:
                for color,points in itertools.zip_longest(colors,data):
                    if points[6] in [0, 5, 8 ]:
                        esa.append("#{0:02x}{1:02x}{2:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255)))
                
    data = np.loadtxt("checkpoints/" + configSettings.EXP_DIR + "/prediction.txt")
    for points in data:
        if not points[6] in [0, 5, 8 ]:
            data.remove(points)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=esa, s=0.01)
    #color_bar = ax.scatter(x, y, z, c=esa, s=0.01)
    #fig.colorbar(color_bar, label='Influence')
    print("Saving plot with GRADCAM...")
    plt.savefig(output_file)




# save_plot_area_3d_real_color()
# save_plot_area_3d_pred_map_color(False)
# save_plot_area_3d_pred_map_color(True)

#save_plot_GRADCAM(4)