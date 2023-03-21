
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

FOLDER_SPECIFIC='ISIC_2019_' # or ''

def main():

    # point to the parent directory that contains the folder 'isic_images'
    # img_data_dir = '/work3/ninwe/dataset/isic/'
    img_data_dir = 'D:\\ninavv\\phd\\data\\isic\\'
    csv_file = '../datafiles/'+FOLDER_SPECIFIC+'metadata-clean-split.csv'

    df = pd.read_csv(csv_file,header=0)

    #df['path_preproc'] = df['path']

    preproc_dir = FOLDER_SPECIFIC+'preproc_224x224/'
    out_dir = img_data_dir

    if not os.path.exists(out_dir + preproc_dir):
        os.makedirs(out_dir + preproc_dir)

    for idx, p in enumerate(tqdm(df['path'])):

        split =  p.split("/")
        preproc_filename = split[-1]
        #df.loc[idx, 'path_preproc'] = preproc_dir + preproc_filename
        out_path = out_dir + preproc_dir + preproc_filename

        if not os.path.exists(out_path):
            image = imread(img_data_dir + p)
            image = resize(image, output_shape=(224, 224), preserve_range=True)
            imsave(out_path, image.astype(np.uint8))

    #df.to_csv(csv_file)


def get_preproc_path():
    # img_data_dir = '/work3/ninwe/dataset/isic/'
    img_data_dir = 'D:\\ninavv\\phd\\data\\isic\\'
    csv_file = '../datafiles/' + FOLDER_SPECIFIC + 'metadata-clean-split.csv'
    df = pd.read_csv(csv_file, header=0)
    df['path_preproc'] = df['path']

    preproc_dir = FOLDER_SPECIFIC + 'preproc_224x224/'
    for idx, p in enumerate(tqdm(df['path'])):
        split = p.split("/")
        preproc_filename = split[-1]
        df.loc[idx, 'path_preproc'] = preproc_dir + preproc_filename
    df.to_csv(csv_file)

if __name__ == '__main__':
    # main()
    get_preproc_path()