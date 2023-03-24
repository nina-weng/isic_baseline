import torchvision.transforms.functional

from prediction.models import DenseNet,ResNet
import os
from dataloader.dataloader import ISICDataModule
from data_preprocess.preprocess import FOLDER_SPECIFIC
from prediction.disease_prediction import image_size,batch_size,num_classes
import torch
from tqdm import tqdm
import torchvision.transforms.functional as F
import torchvision.transforms as T

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# hyperparameters
run_dir = 'D:/ninavv/phd/research/isic_results/disease/'
run_config = 'ISIC_2019_densenet-tp40-lr1e-05-ep50-pt1-aug1'
version_no = 0

checkpoint_dir = run_dir + run_config + '/version_' + str(version_no) + '/checkpoints/'
filenames = os.listdir(checkpoint_dir)
filenames = [each for each in filenames if each.endswith('.ckpt')]

assert len(filenames) == 1, 'more than one checkpoints file in dir'

checkpoint_path = checkpoint_dir + filenames[0]

test_perc =int(run_config.split('-')[1][2:])

model_type = run_config.split('-')[0].split('_')[-1]
model_type = ResNet if model_type=='resnet' else DenseNet

# img_data_dir = '/work3/ninwe/dataset/isic/'
img_data_dir = 'D:/ninavv/phd/data/isic/'
csv_file_img = '../datafiles/'+FOLDER_SPECIFIC+'metadata-clean-split-test{}.csv'.format(test_perc)


def test_counterfactual(model, data_loader, device, counterfactual=False,ct='lighting_all',bf=1.5):
    '''

    :param model:
    :param data_loader:
    :param device:
    :param counterfactual: bool, True or False
    :param ct: refers counterfactual type
        counterfactual operations:
        'lighting_all': adjust the lighting of the whole image
        'lighting_skin': adjust the lighting only on the skin part - segmentation needed in the first place
                #TODO
    :param bf: stands for brightness factor, 0->black, 1->original img, 2->brighter
    :return:
    '''
    model.eval()
    logits = []
    preds = []
    targets = []



    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device) # img shape: (bs,3,224,224)

            if counterfactual:
                if ct=='light_all':
                    img_c = F.adjust_brightness(img,brightness_factor=bf)

                    if index==0:
                        print('print some lighting-changed imgs')
                        for i in range(0, 5):
                            fig, axs = plt.subplots(ncols=2, squeeze=False)
                            axs[0, 0].imshow(img[i].permute(1, 2, 0))  # img[i].permute -> (224,224,3)
                            axs[0, 0].set_title('original img,index:{}'.format(i))
                            axs[0, 0].axis('off')
                            axs[0, 1].imshow(img_c[i].permute(1, 2, 0))
                            axs[0, 1].set_title('{},bf:{},index:{}'.format(ct, bf, i))
                            axs[0, 1].axis('off')
                            plt.show()


                else:
                    raise Exception('Not implemented.')

            if counterfactual:
                img = img_c

            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()



def main(checkpoint_path):
    '''
    counterfactual experiments on lighting setting
    :return:
    '''

    # load pretrained model
    model = model_type.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                            num_classes=num_classes,lr=1e-5,pretrained=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")

    model.to(device)
    model.eval()


    # load dataset
    data = ISICDataModule(img_data_dir=img_data_dir,
                          csv_file_img=csv_file_img,
                          image_size=image_size,
                          pseudo_rgb=False,
                          batch_size=32,
                          num_workers=0,
                          augmentation=True)

    cols_names_classes = ['class_' + str(i) for i in range(0, num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    out_dir = run_dir + run_config


    # config on counterfactual
    counterfactual = True
    ct = 'light_all'
    bf = 0.5

    print('TESTING')
    preds_test, targets_test, logits_test = test_counterfactual(model, data.test_dataloader(), device,
                                                                counterfactual=counterfactual,
                                                                ct=ct,
                                                                bf=bf)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.version_{}.{}-{}.csv'.format(version_no,ct,bf)), index=False)








if __name__ == '__main__':

    main(checkpoint_path)