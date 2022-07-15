import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import configSettings
name_dir = "checkpoints/"+configSettings.EXP_DIR+"/actGradExtractionPlot/actGradExtractionPlotAG_8_new/ag_median_191_tg8.ply"
cloud = o3d.io.read_point_cloud(name_dir)
#cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
points = np.asarray(cloud.points)
colors = None
if cloud.has_colors():
    colors = np.asarray(cloud.colors)
    '''
    if "tg0" in name_dir or "tg3" in name_dir or "tg8" in name_dir:
        for color in colors:
            colors_new=color[0]
            color[0]=color[2]
            color[2]=colors_new
    '''
elif cloud.has_normals():
    colors = (0.5, 0.5, 0.5) + np.asarray(cloud.normals) * 0.5
else:
    geometry.paint_uniform_color((1.0, 0.0, 0.0))
    colors = np.asarray(geometry.colors)


fig = go.Figure(
    data=[
        go.Scatter3d(
            x=points[:,0], y=points[:,1], z=points[:,2],
            mode='markers',
            marker=dict(size=4, color=colors)
        )
    ],
    layout=dict(
        scene=dict(
            xaxis=dict(visible=False, range=[-10,10]),
            yaxis=dict(visible=False, range=[-10,10]),
            zaxis=dict(visible=False, range=[-10,10])
            
            #xaxis=dict(visible=False),
            #yaxis=dict(visible=False),
            #zaxis=dict(visible=False)
            
        )
    )
)
fig.show()