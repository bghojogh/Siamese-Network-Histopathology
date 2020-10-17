import os
import time

import pandas as pd
import numpy as np

import cv2
from skimage.io import imread
from PIL import Image

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import time

NUMBER_OF_TRIPLETS = 1

def uint16(val):
    return np.uint16(val)

def clean_dataframe(dataframe):
    for index, row in dataframe.iterrows():

        anchorUL_idx = ((row['anchor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
        neighborUL_idx = ((row['neighbor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
        distantUL_idx = ((row['distant'].replace('(', '')).replace(')','')).replace(' ','').split(',')
        


        if distantUL_idx == ['None','None'] or neighborUL_idx == ['None','None'] or anchorUL_idx == ['None','None']:
            dataframe = dataframe.drop(index)
            continue
        
        anchorUL_idx = (int(anchorUL_idx[0]),int(anchorUL_idx[1]))
        neighborUL_idx = (int(neighborUL_idx[0]),int(neighborUL_idx[1]))
        distantUL_idx = (int(distantUL_idx[0]),int(distantUL_idx[1]))

        anchor_neighbor = cv2.cvtColor(cv2.imread(row['img_1']), cv2.COLOR_BGR2RGB)
        
        anchor_tile = anchor_neighbor[anchorUL_idx[0]:(anchorUL_idx[0]+128), anchorUL_idx[1]:(anchorUL_idx[1]+128)]

        neighbor_tile = anchor_neighbor[neighborUL_idx[0]:(neighborUL_idx[0]+128), neighborUL_idx[1]:(neighborUL_idx[1]+128)]

        distant = cv2.cvtColor(cv2.imread(row['img_2']), cv2.COLOR_BGR2RGB)
        distant_tile = distant[distantUL_idx[0]:(distantUL_idx[0]+128), distantUL_idx[1]:(distantUL_idx[1]+128)]

        if (anchor_tile.shape != (128,128,3) or neighbor_tile.shape != (128,128,3) or distant_tile.shape != (128,128,3)):
            dataframe = dataframe.drop(index)

    return dataframe


dataframe = pd.read_csv("dataframe-20191122-014750.csv")
dataframe_cleaned = clean_dataframe(dataframe)
file_name = str('cleaned-'+ time.strftime("%Y%m%d-%H%M%S")+'.csv')
dataframe_cleaned.to_csv(file_name)


