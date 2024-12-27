import os

import glob
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import h5py
import cv2
from typing import *
from pathlib import Path

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def load_data(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe

def get_cxr_paths_list(filepath): 
    dataframe = load_data(filepath)
    cxr_paths = dataframe['Path']
    return cxr_paths

'''
This function resizes and zero pads image 
'''
def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    # create a new image and paste the resized on it

    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_img

def img_to_hdf5(cxr_paths: List[Union[str, Path]], out_filepath: str, resolution=320): 
    """
    Convert directory of images into a .h5 file given paths to all 
    images. 
    """
    dset_size = len(cxr_paths)
    failed_images = []
    with h5py.File(out_filepath,'w') as h5f:
        img_dset = h5f.create_dataset('cxr', shape=(dset_size, resolution, resolution))    
        for idx, path in enumerate(tqdm(cxr_paths)):
            try: 
                # read image using cv2
                img = cv2.imread(str(path))
                # convert to PIL Image object
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                # preprocess
                img = preprocess(img_pil, desired_size=resolution)     
                img_dset[idx] = img
            except Exception as e: 
                failed_images.append((path, e))
    print(f"{len(failed_images)} / {len(cxr_paths)} images failed to be added to h5.", failed_images)

def get_files(directory):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file in filenames:
            if file.endswith(".jpg"):
                files.append(os.path.join(dirpath, file))
    return files

def get_cxr_path_csv(out_filepath, directory):
    files = get_files(directory)
    file_dict = {"Path": files}
    df = pd.DataFrame(file_dict)
    df.to_csv(out_filepath, index=False)

def section_start(lines, section=' IMPRESSION'):
    for idx, line in enumerate(lines):
        if line.startswith(section):
            return idx
    return -1

def section_end(lines, section_start):
    num_lines = len(lines)

def getIndexOfLast(l, element):
    """ Get index of last occurence of element
    @param l (list): list of elements
    @param element (string): element to search for
    @returns (int): index of last occurrence of element
    """
    i = max(loc for loc, val in enumerate(l) if val == element)
    return i 

def write_report_csv(cxr_paths, out_path):
    # Initialize the dictionary with placeholder values
    imps = {"filename": [], "impression": []}

    # Iterate over each path in cxr_paths
    for cxr_path in cxr_paths:
        # Instead of extracting values from the text files, use the placeholders
        filename = "null.txt"
        imp = "A photo of a class."

        # Append the placeholders to the dictionary
        imps["filename"].append(filename)
        imps["impression"].append(imp)

    # Convert the dictionary to a DataFrame and save it to a CSV file
    df = pd.DataFrame(data=imps)
    df.to_csv(out_path, index=False)

