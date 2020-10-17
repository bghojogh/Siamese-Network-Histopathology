import os
import time

import pandas as pd
import numpy as np

import cv2
from skimage.io import imread
from PIL import Image

import matplotlib.pyplot as plt


def uint16(val):
    return np.uint16(val)


def create_directory_wrapper(folder_name):
    dir_name = folder_name

    try:
        os.mkdir(dir_name)
        print("Directory " , dir_name ,  " Created ") 

    except FileExistsError:
        print("Directory " , dir_name ,  " already exists")
    
    return os.path.join(os.getcwd(), folder_name)
    

def write_triplets_npy(dataFrame,out_path=os.getcwd()+'triplets.tfrecords'):
    
    dir_triplets_name = create_directory_wrapper('triplets')

    for index, row in dataFrame.iterrows():


        anchorUL_idx = ((row['anchor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
        neighborUL_idx = ((row['neighbor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
        distantUL_idx = ((row['distant'].replace('(', '')).replace(')','')).replace(' ','').split(',')

        anchorUL_idx = (int(anchorUL_idx[0]),int(anchorUL_idx[1]))
        neighborUL_idx = (int(neighborUL_idx[0]),int(neighborUL_idx[1]))
        distantUL_idx = (int(distantUL_idx[0]),int(distantUL_idx[1]))


        anchor_neighbor = cv2.cvtColor(cv2.imread(row['img_1']), cv2.COLOR_BGR2RGB)
        anchor_tile = anchor_neighbor[anchorUL_idx[0]:(anchorUL_idx[0]+128), anchorUL_idx[1]:(anchorUL_idx[1]+128)]

        np.save( dir_triplets_name +'\\'+ str(index)+ 'anchor' + '.npy', anchor_tile)

        neighbor_tile = anchor_neighbor[neighborUL_idx[0]:(neighborUL_idx[0]+128), neighborUL_idx[1]:(neighborUL_idx[1]+128)]
        
        np.save( dir_triplets_name +'\\'+ str(index)+ 'neighbor' + '.npy', neighbor_tile)


        distant = cv2.cvtColor(cv2.imread(row['img_2']), cv2.COLOR_BGR2RGB)
        distant_tile = distant[distantUL_idx[0]:(distantUL_idx[0]+128), distantUL_idx[1]:(distantUL_idx[1]+128)]

        np.save( dir_triplets_name +'\\'+ str(index)+ 'distant' + '.npy', distant_tile)



            
dataframe = pd.read_csv("D:\\triplet_work_Main\\CreateTriplets_npy\\cleaned-20191123-051622.csv")
write_triplets_npy(dataframe)