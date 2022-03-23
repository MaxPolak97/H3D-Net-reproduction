from h3ds.dataset import H3DS
import numpy as np
from tempfile import TemporaryFile
import trimesh
import copy
from ipywidgets import interact, interactive, widgets, fixed
from h3ds.mesh import Mesh
#from vtkplotter import *

h3ds = H3DS(path='h3ds')

scene_id = '3b5a2eb92a501d54'


mesh_pred_path = 'idr_eval_results/reconstructions/idr/' + scene_id + '_3.ply'
mesh_gt_path = 'h3ds_v0.2/' + scene_id + '/full_head.obj'

mesh_pred = Mesh().load(mesh_pred_path)
mesh_gt, images, masks, cameras = h3ds.load_scene('3b5a2eb92a501d54', '3')

vertices = mesh_gt.vertices
vertices_pred = mesh_pred.vertices

closest_vert_right_eye = vertices_pred[0]
closest_vert_left_eye = vertices_pred[0]
closest_vert_nose_base = vertices_pred[0]
closest_vert_right_lips = vertices_pred[0]
closest_vert_left_lips = vertices_pred[0]
closest_vert_nose_tip = vertices_pred[0]

closest_vert = [closest_vert_right_eye,
                closest_vert_left_eye,
                closest_vert_nose_base,
                closest_vert_right_lips,
                closest_vert_left_lips,
                closest_vert_nose_tip]

#print(closest_vert)

#right_eye = [-64.928, -103.996, 552.295]
#left_eye = [-37.282, -105.554, 563.712]
#nose_base = [-47.800, -34.497, 542.937]
#right_lips = [-84.489, 6.352, 558.372]
#left_lips = [-13.259, 8.071, 561.263]
#nose_tip = [-48.228, -65.305, 560.791]

right_eye = [-11.4557/1000, -0.1403, -0.3308]
left_eye = [35.452/1000, -0.15127, -0.315165]
nose_base = [8.871/1000, 20.722/1000, -0.3761]
right_lips = [-84.9446/1000, 0.100465, -0.32450]
left_lips = [77.669/1000, 0.109847, -0.314554]
nose_tip = [13.562/1000, -34.0037/1000, -0.4307]

landmarks = [right_eye, left_eye, nose_base, right_lips, left_lips, nose_tip]


print(landmarks)
print(closest_vert[0])

closest_loss = []

for idx, vertex in enumerate(closest_vert):
    closest_loss.append(
        abs(vertex[0] - landmarks[idx][0]) + abs(vertex[1] - landmarks[idx][1]) + abs(vertex[2] - landmarks[idx][2]))




landmarks_idx = [0, 0, 0, 0, 0, 0]

for idx_1, landmark in enumerate(landmarks):
    for idx, vert in enumerate(vertices_pred):
        loss = abs(vert[0] - landmark[0]) + abs(vert[1] - landmark[1]) + abs(vert[2] - landmark[2])
        if loss < closest_loss[idx_1]:
            closest_vert[idx_1] = vert
            closest_loss[idx_1] = loss
            landmarks_idx[idx_1] = idx

print(closest_loss)

print(landmarks_idx)