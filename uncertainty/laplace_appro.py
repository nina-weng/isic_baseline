from laplace import Laplace

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append('../../isic_baseline')

from prediction.disease_prediction import num_classes,image_size,batch_size
from prediction.models import ResNet,DenseNet
from data_preprocess.preprocess import FOLDER_SPECIFIC
from dataloader.dataloader import ISICDataModule

# hyperparameters for loading the pre-trained model
# run_dir = 'D:/ninavv/phd/research/isic_results/disease/'
run_dir = '/work3/ninwe/run/isic/disease/'
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

img_data_dir = '/work3/ninwe/dataset/isic/'
# img_data_dir = 'D:/ninavv/phd/data/isic/'
csv_file_img = '../datafiles/'+FOLDER_SPECIFIC+'metadata-clean-split-test{}.csv'.format(test_perc)

out_dir = run_dir+run_config+'/laplace_approx_res/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def main(checkpoint_path):
    # Pre-trained model

    # load pretrained model
    model = model_type.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                            num_classes=num_classes, lr=1e-5, pretrained=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")

    model.to(device)
    model.eval()

    # User-specified LA flavor
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='diag')

    # load dataset
    data = ISICDataModule(img_data_dir=img_data_dir,
                          csv_file_img=csv_file_img,
                          image_size=image_size,
                          pseudo_rgb=False,
                          batch_size=32,
                          num_workers=0,
                          augmentation=True)


    print('Start LA Fitting...')
    la.fit(data.train_dataloader())
    print('End of LA Fitting.')

    print('Optimize with Valid Set...')
    la.optimize_prior_precision(method='CV', val_loader=data.val_dataloader())
    print('Finish Optimization.')

    # User-specified predictive approx.
    print('Start testing...')

    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data.test_dataloader(), desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device) # img shape: (bs,3,224,224)

            out = la(img, link_approx='probit')
            print(out.shape)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)


    preds = preds.cpu().numpy()
    targets=targets.cpu().numpy()
    logits = logits.cpu().numpy()

    cols_names_classes = ['class_' + str(i) for i in range(0, num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    df = pd.DataFrame(data=preds, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'laplace_sample_{}.csv'.format(len(os.listdir(out_dir)))), index=False)


if __name__ == '__main__':
    print('start')
    main(checkpoint_path)
