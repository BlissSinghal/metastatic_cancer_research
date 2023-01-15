from pathlib import Path
from typing import List
import imageio 
import os
from os import listdir
import numpy as np

def read_images(image_names, folder_name, path, limit):
    images = []
    for image in image_names:
        image_data = imageio.imread(f"{path}/{folder_name}/{image}")
        images.append(image_data)
    return np.array(images)
def get_image_names(path, folder_name, limit):
    image_names = []
    for image in listdir(f"{path}/{folder_name}"):
        image_names.append(image)
    return image_names[0:limit]
    

def get_training_data(limit):
    path = Path("~/Downloads/cancer_data").expanduser()
    folder = "train"
    image_names = get_image_names(path, folder, limit)
    images = read_images(image_names, folder, path, len(image_names))
    return [image_names, images]