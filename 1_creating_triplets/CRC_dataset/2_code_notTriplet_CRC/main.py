
import numpy as np
# import pandas as pd
from random import shuffle
# from skimage.morphology import binary_erosion
import utils
import time
from PIL import Image
# import PIL
import glob
import matplotlib as plt
import os, os.path
import pandas as pd
import tensorflow as tf


def main():
    crop_again = False
    divide_again = False
    make_numpy_and_tfrecord_again = True
    # path_base = ".\\..\\..\\Kather_texture_2016_dataset\\Kather_texture_2016_image_tiles_5000\\"
    tissue_type_list = ["01_TUMOR", "02_STROMA", "03_COMPLEX", "04_LYMPHO", "05_DEBRIS", "06_MUCOSA", "07_ADIPOSE", "08_EMPTY"]
    path1 = "D:\\Datasets\\Kather_texture_2016_image_tiles_5000\\"
    path2 = "D:\\Datasets\\Kather_texture_2016_image_tiles_5000_cropped\\"
    path3 = "D:\\Datasets\\Kather_train_test\\train\\"
    path4 = "D:\\Datasets\\Kather_train_NotTriplets\\"
    if crop_again:
        crop_images(tissue_type_list, path_base=path1, path_save=path2)
    if divide_again:
        divide_images_to_train_and_test_sets(tissue_type_list, path_base=path2, path_save=path3, train_proportion=0.6)
    if make_numpy_and_tfrecord_again:
        make_numpy_and_tfrecord(tissue_type_list, path_base=path3, path_save_images=path4+"images\\", path_save_types=path4+"types\\", path_save_tfrecord=path4)

def crop_images(tissue_type_list, path_base, path_save, x_start=11, x_end=139, y_start=11, y_end=139):
    for tissue_type in tissue_type_list:
        print("processing tissue: " + tissue_type)
        path = path_base + tissue_type + "\\"
        image_list = read_images(path, image_format="tif")
        for i, image_ in enumerate(image_list):
            imgarr = np.array(image_)
            imgarr = imgarr[y_start:y_end, x_start:x_end, :]
            image_list[i] = imgarr
            save_numpy(path=path_save+tissue_type+"\\", name_=str(i), array_=imgarr)
    
def divide_images_to_train_and_test_sets(tissue_type_list, path_base, path_save, train_proportion=0.6):
    for tissue_type in tissue_type_list:
        print("processing tissue: " + tissue_type)
        path_ = path_base + tissue_type + "\\"
        count_ = count_files_in_folder(path_)
        shuffled_indices = np.random.permutation(int(count_))
        n_train_set = int(train_proportion * len(shuffled_indices))
        for file_name in shuffled_indices[:n_train_set]:
            file_ = np.load(path_+str(file_name)+".npy")
            save_numpy(path=path_save+"train\\"+tissue_type+"\\", name_=str(file_name), array_=file_)
        for file_name in shuffled_indices[n_train_set:]:
            file_ = np.load(path_+str(file_name)+".npy")
            save_numpy(path=path_save+"test\\"+tissue_type+"\\", name_=str(file_name), array_=file_)
    
def make_numpy_and_tfrecord(tissue_type_list, path_base, path_save_images, path_save_types, path_save_tfrecord, tfrecord_name='images.tfrecords'):
    paths_of_train_images = glob.glob(path_base+"**\\*.npy")
    n_images = len(paths_of_train_images)
    if not os.path.exists(path_save_tfrecord):
        os.makedirs(path_save_tfrecord)
    with tf.io.TFRecordWriter(path_save_tfrecord + tfrecord_name) as writer:
        for image_index in range(n_images):
            if image_index % 20 == 0:
                print("processing image " + str(image_index) + "....")
            image_path = paths_of_train_images[image_index]
            save_image_as_numpy(image_path, path_save_images, path_save_types, image_index, tissue_type_list)
            save_image_as_tfrecord(image_path, path_save_images, path_save_types, image_index, tissue_type_list, writer)

def save_image_as_numpy(image_path, path_save_images, path_save_types, image_index, tissue_type_list):
    image_npy = np.load(image_path)
    save_numpy(path=path_save_images, name_=str(image_index), array_=image_npy)
    type_of_image = image_path.split("\\")[-2]
    type_of_image_key = tissue_type_list.index(type_of_image) + 1  #--> encode type as number (starting from 1)
    save_numpy(path=path_save_types, name_=str(image_index)+"_type", array_=np.array([type_of_image_key]))

def save_image_as_tfrecord(image_path, path_save_images, path_save_types, image_index, tissue_type_list, writer):
    # anchor:
    image_npy = np.load(image_path)
    image_tile_bytes = image_npy.tostring()
    type_of_image = image_path.split("\\")[-2]
    type_of_image_key = tissue_type_list.index(type_of_image) + 1  #--> encode type as number (starting from 1)
    # wrapping:
    data = \
        {
            'image': wrap_bytes(image_tile_bytes),
            'tissue_type': wrap_int64(type_of_image_key)
        }
    feature = tf.train.Features(feature=data)
    example = tf.train.Example(features=feature)
    serialized = example.SerializeToString()
    writer.write(serialized)

def count_files_in_folder(path_):
    # simple version for working with CWD
    return len([name for name in os.listdir(path_) if os.path.isfile(os.path.join(path_, name))])

def save_numpy(path, name_, array_):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+name_+".npy", array_)

def read_images(path, image_format="tif"):
    image_list = []
    for filename in glob.glob(path+"*."+image_format):
        # print("processing file: " + str(filename))
        im = Image.open(filename)
        image_list.append(im)
    return image_list

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == "__main__":
    main()