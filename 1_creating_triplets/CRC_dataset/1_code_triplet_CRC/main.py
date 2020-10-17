
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
    make_triplet_again = True
    n_triplets = 22528
    # path_base = ".\\..\\..\\Kather_texture_2016_dataset\\Kather_texture_2016_image_tiles_5000\\"
    tissue_type_list = ["01_TUMOR", "02_STROMA", "03_COMPLEX", "04_LYMPHO", "05_DEBRIS", "06_MUCOSA", "07_ADIPOSE", "08_EMPTY"]
    path1 = "D:\\Datasets\\Kather_texture_2016_image_tiles_5000\\"
    path2 = "D:\\Datasets\\Kather_texture_2016_image_tiles_5000_cropped\\"
    path3 = "D:\\Datasets\\Kather_train_test\\"
    path4 = "D:\\Datasets\\Kather_train_triplets\\"
    if crop_again:
        crop_images(tissue_type_list, path_base=path1, path_save=path2)
    if divide_again:
        divide_images_to_train_and_test_sets(tissue_type_list, path_base=path2, path_save=path3, train_proportion=0.6)
    if make_triplet_again:
        make_triplets(tissue_type_list, path_base=path3, path_save_images=path4+"triplets\\", path_save_types=path4+"types\\", path_save_tfrecord=path4, n_triplets=n_triplets)

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
    
def make_triplets(tissue_type_list, path_base, path_save_images, path_save_types, path_save_tfrecord, n_triplets, tfrecord_name='triplets.tfrecords'):
    df = create_dataframe(tissue_type_list, path_base)
    if not os.path.exists(path_save_tfrecord):
        os.makedirs(path_save_tfrecord)
    with tf.io.TFRecordWriter(path_save_tfrecord + tfrecord_name) as writer:
        for triplet_index in range(n_triplets):
            if triplet_index % 20 == 0:
                print("processing triplet " + str(triplet_index) + "....")
            anchor_path, neighbor_path, distant_path = extract_one_triplet(tissue_type_list, df)
            save_triplets_as_numpy(anchor_path, neighbor_path, distant_path, path_save_images, path_save_types, triplet_index, tissue_type_list)
            save_triplets_as_tfrecord(anchor_path, neighbor_path, distant_path, path_save_images, path_save_types, triplet_index, tissue_type_list, writer)

def create_dataframe(tissue_type_list, path_base):
    path_ = path_base + "train\\**\\"
    file_names = glob.glob(path_+"*.npy")
    type_list = [file_name.split("\\")[-2] for file_name in file_names]
    df = pd.DataFrame({"path":file_names})
    df.insert(1, "type", type_list, True)
    return df

def extract_one_triplet(tissue_type_list, df):
    n_triplets = len(tissue_type_list)
    temp = np.random.choice(np.array([i for i in range(n_triplets)]), size=2, replace=False)
    anchor_neighbor_type = tissue_type_list[temp[0]]
    distant_type = tissue_type_list[temp[1]]
    rows_of_anchors_and_neighbors = df.loc[df['type'] == anchor_neighbor_type]
    rows_of_distants = df.loc[df['type'] == distant_type]
    anchor_and_neighbor_rows_df = rows_of_anchors_and_neighbors.sample(n=2, random_state=1, replace=False)
    distant_row_df = rows_of_distants.sample(n=1, random_state=1)
    anchor_path = anchor_and_neighbor_rows_df.loc[:, "path"].values.tolist()[0]
    neighbor_path = anchor_and_neighbor_rows_df.loc[:, "path"].values.tolist()[1]
    distant_path = distant_row_df.loc[:, "path"].values.tolist()[0]
    return anchor_path, neighbor_path, distant_path

def save_triplets_as_numpy(anchor_path, neighbor_path, distant_path, path_save_images, path_save_types, triplet_index, tissue_type_list):
    # anchor:
    anchor_npy = np.load(anchor_path)
    save_numpy(path=path_save_images, name_=str(triplet_index)+"anchor", array_=anchor_npy)
    type_of_anchor = anchor_path.split("\\")[-2]
    type_of_anchor_key = tissue_type_list.index(type_of_anchor) + 1  #--> encode type as number (starting from 1)
    save_numpy(path=path_save_types, name_=str(triplet_index)+"anchor_type", array_=np.array([type_of_anchor_key]))
    # neighbor:
    neighbor_npy = np.load(neighbor_path)
    save_numpy(path=path_save_images, name_=str(triplet_index)+"neighbor", array_=neighbor_npy)
    type_of_neighbor = neighbor_path.split("\\")[-2]
    type_of_neighbor_key = tissue_type_list.index(type_of_neighbor) + 1  #--> encode type as number (starting from 1)
    save_numpy(path=path_save_types, name_=str(triplet_index)+"neighbor_type", array_=np.array([type_of_neighbor_key]))
    # distant:
    distant_npy = np.load(distant_path)
    save_numpy(path=path_save_images, name_=str(triplet_index)+"distant", array_=distant_npy)
    type_of_distant = distant_path.split("\\")[-2]
    type_of_distant_key = tissue_type_list.index(type_of_distant) + 1  #--> encode type as number (starting from 1)
    save_numpy(path=path_save_types, name_=str(triplet_index)+"distant_type", array_=np.array([type_of_distant_key]))

def save_triplets_as_tfrecord(anchor_path, neighbor_path, distant_path, path_save_images, path_save_types, triplet_index, tissue_type_list, writer):
    # anchor:
    anchor_npy = np.load(anchor_path)
    anchor_tile_bytes = anchor_npy.tostring()
    type_of_anchor = anchor_path.split("\\")[-2]
    type_of_anchor_key = tissue_type_list.index(type_of_anchor) + 1  #--> encode type as number (starting from 1)
    # neighbor:
    neighbor_npy = np.load(neighbor_path)
    neighbor_tile_bytes = neighbor_npy.tostring()
    type_of_neighbor = neighbor_path.split("\\")[-2]
    type_of_neighbor_key = tissue_type_list.index(type_of_neighbor) + 1  #--> encode type as number (starting from 1)
    # distant:
    distant_npy = np.load(distant_path)
    distant_tile_bytes = distant_npy.tostring()
    type_of_distant = distant_path.split("\\")[-2]
    type_of_distant_key = tissue_type_list.index(type_of_distant) + 1  #--> encode type as number (starting from 1)
    # wrapping:
    data = \
        {
            'image_anchor': wrap_bytes(anchor_tile_bytes),
            'image_neighbor': wrap_bytes(neighbor_tile_bytes),
            'image_distant': wrap_bytes(distant_tile_bytes),
            'tissue_type_anchor': wrap_int64(type_of_anchor_key),
            'tissue_type_neighbor': wrap_int64(type_of_neighbor_key),
            'tissue_type_distant': wrap_int64(type_of_distant_key)
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