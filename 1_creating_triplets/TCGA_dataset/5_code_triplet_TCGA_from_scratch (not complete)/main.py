
import os
import time
import pandas as pd
import numpy as np
import cv2
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
import utils1
import glob
from random import shuffle
from skimage.morphology import binary_erosion


def main():
    pass

def take_df_from_web():
    fields = ["file_name",
          "cases.case_id",
          "cases.primary_site",
          "cases.diagnoses.primary_diagnosis",
          "cases.project.disease_type",
          "cases.samples.is_ffpe",
          "md5sum",
          "file_size",
          "state",
          "cases.diagnoses.tissue_or_organ_of_origin",
          "cases.diagnoses.morphology",
          "experimental_strategy"]

    fields = ",".join(fields)

    files_endpt = "https://api.gdc.cancer.gov/files"

    # This set of filters is nested under an 'and' operator.
    filters = {
        "op": "and",
        "content":[
            {
            "op": "in",
            "content":{
                "field": "cases.project.primary_site",
                "value": ["Lung","Colorectal", "Prostate"]
                }
            },
            {
            "op": "in",
            "content":{
                "field": "files.data_format",
                "value": ["SVS"]
                }
            }
        ]
    }
    # filters = {
    #        "op": "=",
    #        "content":{
    #             "field": "files.data_format",
    #             "value": ["SVS"]
    #        }
    #     }

    # A POST is used, so the filter parameters can be passed directly as a Dict object.
    params = {
        "filters": filters,
        "fields": fields,
        "format": "CSV",
        "size": "50000"
        }

    # The parameters are passed to 'json' rather than 'params' in this case
    response = requests.post(files_endpt, headers = {"Content-Type": "application/json"}, json = params)





if __name__ == "__main__":
    main()