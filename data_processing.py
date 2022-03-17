# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:59:04 2022

@author: Maxpo
"""

from h3ds.dataset import H3DS
import os

# Load dataset
h3ds = H3DS(path='h3ds_v0.2')


# Check different views
views_configs = h3ds.default_views_configs(scene_id='0cd3f3c0bc34a287') # '3', '4', '8', '16' and '32'

# Load data of given scene and view configuration
_, images, masks, cameras = h3ds.load_scene(scene_id='0cd3f3c0bc34a287', views_config_id='8', normalized=False)


folder = 'scan1'

# Creates a directory to save the data
if not os.path.isdir(folder):
    os.mkdir(folder) 

# Creates a directory to save the data
if not os.path.isdir(folder + '/image'):
    os.mkdir(folder + '/image') 

# Save images
for n, image in enumerate(images):
    image.save(folder + '/image/img_000' + str(n) + '.jpg')
    
# Creates a directory to save the data
if not os.path.isdir(folder + '/mask'):
    os.mkdir(folder + '/mask') 

# Save masks
for n, mask in enumerate(masks):
    mask.save(folder + '/mask/mask_000' + str(n) + '.jpg')
    

    
    