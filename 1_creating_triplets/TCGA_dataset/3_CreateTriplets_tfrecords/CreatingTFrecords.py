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


def uint16(val):
    return np.uint16(val)

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_directory_wrapper(folder_name):
    dir_name = folder_name

    try:
        os.mkdir(dir_name)
        print("Directory " , dir_name ,  " Created ") 

    except FileExistsError:
        print("Directory " , dir_name ,  " already exists")
    
    return os.path.join(os.getcwd(), folder_name)
    

def convert_to_tfrecords(data_frame, tfrecord_name = 'triplets.tfrecords'):
    
    dir_triplets_name = create_directory_wrapper('tfrecords')

    with tf.io.TFRecordWriter(dir_triplets_name + '//'+ tfrecord_name) as writer:

        for index, row in data_frame.iterrows():

            anchorUL_idx = ((row['anchor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
            neighborUL_idx = ((row['neighbor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
            distantUL_idx = ((row['distant'].replace('(', '')).replace(')','')).replace(' ','').split(',')

            anchorUL_idx = (int(anchorUL_idx[0]),int(anchorUL_idx[1]))
            neighborUL_idx = (int(neighborUL_idx[0]),int(neighborUL_idx[1]))
            distantUL_idx = (int(distantUL_idx[0]),int(distantUL_idx[1]))


            anchor_neighbor = cv2.cvtColor(cv2.imread(row['img_1']), cv2.COLOR_BGR2RGB)
            anchor_tile = anchor_neighbor[anchorUL_idx[0]:(anchorUL_idx[0]+128), anchorUL_idx[1]:(anchorUL_idx[1]+128)]
            anchor_tile_bytes = anchor_tile.tostring()



            neighbor_tile = anchor_neighbor[neighborUL_idx[0]:(neighborUL_idx[0]+128), neighborUL_idx[1]:(neighborUL_idx[1]+128)]
            neighbor_tile_bytes = neighbor_tile.tostring()



            distant = cv2.cvtColor(cv2.imread(row['img_2']), cv2.COLOR_BGR2RGB)
            distant_tile = distant[distantUL_idx[0]:(distantUL_idx[0]+128), distantUL_idx[1]:(distantUL_idx[1]+128)]
            distant_tile_bytes = distant_tile.tostring()

            data = \
               {
                   'image_anchor': wrap_bytes(anchor_tile_bytes),
                   'image_neighbor': wrap_bytes(neighbor_tile_bytes),
                   'image_distant': wrap_bytes(distant_tile_bytes)
               }

            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            writer.write(serialized)



            
data_frame = pd.read_csv("D:\\triplet_work_Main\\_Important_files\\cleaned-20191123-051622.csv")
convert_to_tfrecords(data_frame)