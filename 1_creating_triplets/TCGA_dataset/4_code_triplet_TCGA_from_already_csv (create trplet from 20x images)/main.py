
import os
import pandas as pd
import numpy as np
import openslide
from PIL import Image
import tensorflow as tf



def main():
    update_df_again = False
    extract_triplets_again = True
    cleaned_df_path = ".\\cleaned-20191123-051622.csv"
    final_df_path = ".\\final_dataframe.csv"
    final_df_path_save = ".\\final_dataframe_.csv"
    encoding_file_path_save_type = ".\\encoding_file_type.txt"
    encoding_file_path_save_subtype = ".\\encoding_file_subtype.txt"
    path_save_tfrecord = ".\\"
    types_list = ["Lung", "Gastrointestinal", "Prostate"]
    if update_df_again:
        cleaned_df = pd.read_csv(cleaned_df_path)
        final_df = pd.read_csv(final_df_path)
        type_list_anchor, subtype_list_anchor, type_list_distant, subtype_list_distant = extract_types_and_subtypes(cleaned_df)
        final_df_ = add_types_and_subtypes_columns(final_df, type_list_anchor, subtype_list_anchor, type_list_distant, subtype_list_distant)
        final_df_.to_csv(final_df_path_save)
    if extract_triplets_again:
        final_df_ = pd.read_csv(final_df_path_save)
        write_triplets(final_df_, types_list, encoding_file_path_save_type, encoding_file_path_save_subtype, path_save_tfrecord)



def extract_types_and_subtypes(cleaned_df_):
    # img_1:
    list_of_paths_1 = cleaned_df_.loc[:, "img_1"]
    list_of_paths_1 = [file_name.replace("//", "/") for file_name in list_of_paths_1]
    list_of_paths_1 = [file_name.replace("/", "\\") for file_name in list_of_paths_1]
    type_list_anchor = [file_name.split("\\")[-4] for file_name in list_of_paths_1]
    subtype_list_anchor = [file_name.split("_")[-1] for file_name in list_of_paths_1]
    subtype_list_anchor = [file_name.split(".")[0] for file_name in subtype_list_anchor]
    # img_2:
    list_of_paths_2 = cleaned_df_.loc[:, "img_2"]
    list_of_paths_2 = [file_name.replace("//", "/") for file_name in list_of_paths_2]
    list_of_paths_2 = [file_name.replace("/", "\\") for file_name in list_of_paths_2]
    type_list_distant = [file_name.split("\\")[-4] for file_name in list_of_paths_2]
    subtype_list_distant = [file_name.split("_")[-1] for file_name in list_of_paths_2]
    subtype_list_distant = [file_name.split(".")[0] for file_name in subtype_list_distant]
    return type_list_anchor, subtype_list_anchor, type_list_distant, subtype_list_distant

def add_types_and_subtypes_columns(final_df_, type_list_anchor, subtype_list_anchor, type_list_distant, subtype_list_distant):
    final_df_.insert(loc=6, column="anchor_type", value=type_list_anchor)
    final_df_.insert(loc=7, column="anchor_subtype", value=subtype_list_anchor)

    final_df_.insert(loc=8, column="neighbor_type", value=type_list_anchor)
    final_df_.insert(loc=9, column="neighbor_subtype", value=subtype_list_anchor)

    final_df_.insert(loc=10, column="distant_type", value=type_list_distant)
    final_df_.insert(loc=11, column="distant_subtype", value=subtype_list_distant)

    return final_df_

def write_triplets(dataFrame, types_list, encoding_file_path_save_type, encoding_file_path_save_subtype, path_save_tfrecord, tfrecord_name='triplets.tfrecords'):
    
    subtypes_of_types = encode_subtypes_beforehand(types_list, dataFrame, encoding_file_path_save_type, encoding_file_path_save_subtype)
    if not os.path.exists(path_save_tfrecord):
        os.makedirs(path_save_tfrecord)
    with tf.io.TFRecordWriter(path_save_tfrecord + tfrecord_name) as writer:
        for index, row in dataFrame.iterrows():

            anchorUL_idx = ((row['anchor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
            neighborUL_idx = ((row['neighbor'].replace('(', '')).replace(')','')).replace(' ','').split(',')
            distantUL_idx = ((row['distant'].replace('(', '')).replace(')','')).replace(' ','').split(',')

            anchorUL_idx = (int(anchorUL_idx[0]),int(anchorUL_idx[1]))
            neighborUL_idx = (int(neighborUL_idx[0]),int(neighborUL_idx[1]))
            distantUL_idx = (int(distantUL_idx[0]),int(distantUL_idx[1]))
            
            # anchor and neighbor:
            anchor_neighbor = openslide.OpenSlide(row['img_1'])
            power_anchor_neighbor = int(anchor_neighbor.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            if power_anchor_neighbor == 40:
                anchor_tile = anchor_neighbor.read_region((anchorUL_idx[0],anchorUL_idx[1]),0, (256,256))
                neighbor_tile = anchor_neighbor.read_region((neighborUL_idx[0],neighborUL_idx[1]),0, (256,256))
            else:
                anchor_tile = anchor_neighbor.read_region((anchorUL_idx[0],anchorUL_idx[1]),0, (128,128))
                neighbor_tile = anchor_neighbor.read_region((neighborUL_idx[0],neighborUL_idx[1]),0, (128,128))
            anchor_tile = anchor_tile.resize((128,128), resample=PIL.Image.LANCZOS)
            neighbor_tile = neighbor_tile.resize((128,128), resample=PIL.Image.LANCZOS)
            np.save( path_save_tfrecord +'\\triplets\\'+ str(index)+ 'anchor' + '.npy', anchor_tile)
            np.save( path_save_tfrecord +'\\triplets\\'+ str(index)+ 'neighbor' + '.npy', neighbor_tile)
            # anchor type and subtype:
            type_of_anchor = row.loc[:, "anchor_type"]
            subtype_of_anchor = row.loc[:, "anchor_subtype"]
            type_of_anchor, subtype_of_anchor = encode_type_and_subtype(type_of_anchor, subtype_of_anchor, subtypes_of_types, types_list)
            np.save(path_save_tfrecord +'\\types_subtypes\\'+ str(index)+ 'anchor_type' + '.npy', np.array([type_of_anchor]))
            np.save(path_save_tfrecord +'\\types_subtypes\\'+ str(index)+ 'anchor_subtype' + '.npy', np.array([subtype_of_anchor]))
            # neighbor type and subtype:
            type_of_neighbor = row.loc[:, "neighbor_type"]
            subtype_of_neighbor = row.loc[:, "neighbor_subtype"]
            type_of_neighbor, subtype_of_neighbor = encode_type_and_subtype(type_of_neighbor, subtype_of_neighbor, subtypes_of_types, types_list)
            np.save(path_save_tfrecord +'\\types_subtypes\\'+ str(index)+ 'neighbor_type' + '.npy', np.array([type_of_neighbor]))
            np.save(path_save_tfrecord +'\\types_subtypes\\'+ str(index)+ 'neighbor_subtype' + '.npy', np.array([subtype_of_neighbor]))

            # distant:
            distant = openslide.OpenSlide(row['img_2'])
            power_distant = int(distant.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            if power_distant == 40:
                distant_tile = distant.read_region((distantUL_idx[0],distantUL_idx[1]),0, (256,256))
            else:
                distant_tile = distant.read_region((distantUL_idx[0],distantUL_idx[1]),0, (128,128))
            distant_tile = distant_tile.resize((128,128), resample=PIL.Image.LANCZOS) 
            np.save( path_save_tfrecord +'\\triplets\\'+ str(index)+ 'distant' + '.npy', distant_tile)
            # distant type and subtype:
            type_of_distant = row.loc[:, "distant_type"]
            subtype_of_distant = row.loc[:, "distant_subtype"]
            type_of_distant, subtype_of_distant = encode_type_and_subtype(type_of_distant, subtype_of_distant, subtypes_of_types, types_list)
            np.save(path_save_tfrecord +'\\types_subtypes\\'+ str(index)+ 'distant_type' + '.npy', np.array([type_of_distant]))
            np.save(path_save_tfrecord +'\\types_subtypes\\'+ str(index)+ 'distant_subtype' + '.npy', np.array([subtype_of_distant]))

            # TFRECORD creation:
            anchor_tile_bytes = anchor_tile.tostring()
            neighbor_tile_bytes = neighbor_tile.tostring()
            distant_tile_bytes = distant_tile.tostring()
            # wrapping:
            data = \
                {
                    'image_anchor': wrap_bytes(anchor_tile_bytes),
                    'image_neighbor': wrap_bytes(neighbor_tile_bytes),
                    'image_distant': wrap_bytes(distant_tile_bytes),
                    'type_anchor': wrap_int64(type_of_anchor),
                    'type_neighbor': wrap_int64(type_of_neighbor),
                    'type_distant': wrap_int64(type_of_distant),
                    'subtype_anchor': wrap_int64(subtype_of_anchor),
                    'subtype_neighbor': wrap_int64(subtype_of_neighbor),
                    'subtype_distant': wrap_int64(subtype_of_distant)
                }
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            writer.write(serialized)

def encode_type_and_subtype(type_, subtype_, subtypes_of_types, types_list):
    type_index = types_list.index(type_)
    subtypes_of_type = subtypes_of_types[type_index]
    subtype_index = subtypes_of_type.index(subtype_)
    return type_index, subtype_index

def encode_subtypes_beforehand(types_list, df, encoding_file_path_save_type, encoding_file_path_save_subtype):
    subtypes_of_types = [None] * len(types_list)
    for type_index, type_ in enumerate(types_list):
        print(type_)
        df_anchor_type = df.loc[df['anchor_type'] == type_]
        df_neighbor_type = df.loc[df['neighbor_type'] == type_]
        df_distant_type = df.loc[df['distant_type'] == type_]
        subtypes_anchor = list(df_anchor_type.anchor_subtype.unique())
        subtypes_neighbor = list(df_neighbor_type.neighbor_subtype.unique())
        subtypes_distant = list(df_distant_type.distant_subtype.unique())
        temp = list(set(subtypes_anchor).union(subtypes_neighbor))
        subtypes_of_this_type = list(set(temp).union(subtypes_distant))  #--> union of subtypes_anchor, subtypes_neighbor, and subtypes_distant
        subtypes_of_types[type_index] = subtypes_of_this_type
        with open(encoding_file_path_save_type, "w") as output:
            output.write(str(types_list))
        with open(encoding_file_path_save_subtype, "w") as output:
            output.write(str(subtypes_of_types))
    return subtypes_of_types

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    # types1 = df.anchor_type.unique().to_list()
    # types2 = df.anchor_type.unique().to_list()
    # types3 = df.anchor_type.unique().to_list()
    # temp1 = list(set(types1).union(types2))
    # types = list(set(temp1).union(types3))  #--> union of types1, types2, and types3

    # if type_ == "Lung":
    #     df
    # elif type_ == "Gastrointestinal":
    #     type_ = 1
    # elif type_ == "Prostate":
    #     type_ = 2
    # return type_

def create_directory_wrapper(folder_name):
    dir_name = folder_name
    try:
        os.mkdir(dir_name)
        print("Directory " , dir_name ,  " Created ") 
    except FileExistsError:
        print("Directory " , dir_name ,  " already exists")
    return os.path.join(os.getcwd(), folder_name)

if __name__ == "__main__":
    main()