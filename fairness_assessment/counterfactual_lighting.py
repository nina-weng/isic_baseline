from prediction.models import DenseNet,ResNet
import os
from dataloader.dataloader import ISICDataModule
from data_preprocess.preprocess import FOLDER_SPECIFIC
from prediction.disease_prediction import image_size,batch_size,args,num_classes
import torch
from tqdm import tqdm



# hyperparameters
run_dir = 'D:/ninavv/phd/research/isic_results/disease/'
run_config = 'densenet-tp40-lr1e-05-ep50-pt1-aug1'
version_no = 6

checkpoint_dir = run_dir + run_config + '/version_' + str(version_no) + '/checkpoints/'
filenames = os.listdir(checkpoint_dir)
filenames = [each if each.endswith('.ckpt') for each in filenames]

assert len(filenames) == 1, 'more than one checkpoints file in dir'

checkpoint_path = checkpoint_dir + filenames[0]

test_perc =int(run_config.split('-')[1][:2])

# img_data_dir = '/work3/ninwe/dataset/isic/'
img_data_dir = 'D:/ninavv/phd/data/isic/'
csv_file_img = '../datafiles/'+FOLDER_SPECIFIC+'metadata-clean-split-test{}.csv'.format(test_perc)


def test_counterfactual(model, data_loader, device, counterfactual='lighting_overall'):
    '''

    :param model:
    :param data_loader:
    :param device:
    :param counterfactual:
        counterfactual operations:
        'lighting': do lighting augments on image

    :return:
    '''
    model.eval()
    logits = []
    preds = []
    targets = []



    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)

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

    info_ = checkpoint_path.split('/')
    run_config = info_[-3]
    print('run_config:{}'.format(run_config))

    model_type = run_config.split('-')[0]
    model_type = DenseNet if model_type=='densenet' else ResNet

    # load pretrained model
    model = model_type.load_from_checkpoint(checkpoint_path=checkpoint_path)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.dev) if use_cuda else "cpu")

    model.to(device)
    model.eval()


    # load dataset
    data = ISICDataModule(img_data_dir=img_data_dir,
                          csv_file_img=csv_file_img,
                          image_size=image_size,
                          pseudo_rgb=False,
                          batch_size=batch_size,
                          num_workers=0,
                          augmentation=False)








if __name__ == '__main__':

    main(checkpoint_path)