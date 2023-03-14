import os
import pandas as pd
import cv2
from PIL import Image
import math
from skimage import io, color
import torch

image_size=(224,224)

# Hair removal for ITA calculation
def hair_remove(image):
    # Convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return final_image


# Calculates Fitzpatrick skin type of an image using Kinyanjui et al.'s thresholds
def get_sample_ita_kin(path):
    ita_bnd_kin = -1
    try:
        rgb = io.imread(path)
        rgb = hair_remove(rgb)
        lab = color.rgb2lab(rgb)
        ita_lst = []
        ita_bnd_lst = []

        xloc_start = [ 230,5,115,216,216,20,20]
        yloc_start = [115,115,5,230,216,20,216]

        scale=image_size[0]/256

        xloc_start = [int(scale*x) for x in xloc_start]
        yloc_start = [int(scale * y) for y in yloc_start]

        interval = 20

        L_lst=[]
        b_lst=[]

        for i in range(7):
            # Taking samples from different parts of the image
            L_lst.append(lab[xloc_start[i]:xloc_start[i]+interval, yloc_start[i]:yloc_start[i]+interval, 0].mean())
            b_lst.append(lab[xloc_start[i]:xloc_start[i] + interval, yloc_start[i]:yloc_start[i] + interval, 2].mean())


        # Calculating ITA values
        for L, b in zip(L_lst, b_lst):
            ita = math.atan((L - 50) / b) * (180 / math.pi)
            ita_lst.append(ita)

        # Using max ITA value (lightest)
        ita_max = max(ita_lst)

        # Getting skin shade band from ITA
        if ita_max > 55:
            ita_bnd_kin = 1
        if 41 < ita_max <= 55:
            ita_bnd_kin = 2
        if 28 < ita_max <= 41:
            ita_bnd_kin = 3
        if 19 < ita_max <= 28:
            ita_bnd_kin = 4
        if 10 < ita_max <= 19:
            ita_bnd_kin = 5
        if ita_max <= 10:
            ita_bnd_kin = 6
    except Exception:
        pass

    return ita_bnd_kin

def get_isic_skin_type():
    # Getting ITA for ISIC Training and saving to csv
    df_train = pd.read_csv('../datafiles/ISIC_2019_metadata-clean-split-test40.csv')

    # data_dir = 'D:/ninavv/phd/data/isic/'
    data_dir = '/work3/ninwe/dataset/isic/'

    df_train['fitzpatrick'] = df_train['path_preproc'].apply(lambda x: get_sample_ita_kin(data_dir+x))
    df_train.to_csv('../datafiles/ISIC_2019_metadata-clean-split-test40-skintone.csv', index=False)


if __name__ == '__main__':
    get_isic_skin_type()
