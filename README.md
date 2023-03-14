## Fairness analysis on ISIC using resnet

**Acknowledgement**: 
The development of this repository is based on Glock, Jones, Bernhardt and Winzeck's work ([their github repo](https://github.com/biomedia-mira/chexploration)).


### Dataset:
The dataset being used is from [ISIC challenge 2019](https://challenge.isic-archive.com/data/#2019). 
Since the ground truth labels are not public, we split the training data they provided into training, validation and testing.
The sex and approximate age information are also offered by the challenge in metadata csv file.

### Requirement:
TODO


### To Run the Experiment:
Before running the experiment:
1. Preprocess the csv file:
run `./notebooks/csv_preprocess.ipynb` to get the clean and formatted csv file for training, which also includes split information.
2. Preprocess the images:
run `./data_preprocess/preprocess.py` to get pre-processed images. This step might take 1-2 hours.

To run the experiment:
`python3 ./prediction/disease_prediction.py`

### Fairness Analysis
Check `./fairness_assessment/bais_analysis.py` for bias analysis on both prediction performance and latent space.





