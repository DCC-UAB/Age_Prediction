# coding: utf-8

#############################################
# Consistent Cumulative Logits with ResNet-34
#############################################

# Imports

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
from AgePrediction.utils.utils import *
from AgePrediction.models.models import ResNetCoral, BasicBlock

torch.backends.cudnn.deterministic = True

CLASSE_CSV_PATH = './coral-cnn-master/datasets/cacd_test.csv'
IMAGE_PATH = './coral-cnn-master/datasets/CACD2000-centered'

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',
                    type=int,
                    default=-1)

parser.add_argument('-s', '--state_dict_path',
                    type=str,
                    required=True)

parser.add_argument('--outpath',
                    type=str,
                    required=True)

parser.add_argument('-d', '--dataset',
                    help="Options: 'afad', 'morph2', or 'cacd'.",
                    type=str,
                    required=True)

args = parser.parse_args()

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")


PATH = args.outpath
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')
STATE_DICT_PATH = args.state_dict_path
GRAYSCALE = False

df = pd.read_csv(CLASSE_CSV_PATH, index_col=0)
ages = df['age'].values
del df
ages = torch.tensor(ages, dtype=torch.float)

if args.dataset == 'afad':
    NUM_CLASSES = 26
    ADD_CLASS = 15

elif args.dataset == 'cacd':
    NUM_CLASSES = 49
    ADD_CLASS = 14

else:
    raise ValueError("args.dataset must be 'afad',"
                     " or 'cacd'. Got %s " % (args.dataset))

BATCH_SIZE = 256
NUM_WORKERS = 6

custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])

classe_dataset = CACDDataset(csv_path=CLASSE_CSV_PATH,
                           img_dir=IMAGE_PATH,
                           transform=custom_transform2)

test_loader = DataLoader(dataset=classe_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNetCoral(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model
#######################
### Initialize Model
#######################

model = resnet34(NUM_CLASSES, GRAYSCALE)
model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=DEVICE))
model.eval()

########## SAVE PREDICTIONS ######
all_pred_str = []
all_pred_int = []
all_probas = []

with torch.set_grad_enabled(False):
    for batch_idx, (features, targets, levels) in enumerate(test_loader):
        lst_str = []
        lst_int = []
        # features = features.to(DEVICE)
        logits, probas = model(features)
        all_probas.append(probas)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        for i in (predicted_labels):
            lst_str.append(str(int(i)))
            lst_int.append(int(i))
        all_pred_str.extend(lst_str)
        all_pred_int.extend(lst_int)

all_pred_int = torch.tensor(all_pred_int, dtype=torch.int)
dif = ages - all_pred_int

for i in range(len(dif)):
    print("Pred:", int(all_pred_int[i]), "Age:", int(ages[i]), "Dif:", int(dif[i]))

print("\nmitjana dif:")
print(torch.mean(dif.float()))
print("\nmitjana abs(dif):")
print(torch.mean(torch.abs(dif.float())))
print("\nstd:")
print(torch.std(dif.float()))
print("\nmin:")
print(torch.min(dif.float()))
print("\nmax:")
print(torch.max(dif.float()))


torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
with open(TEST_PREDICTIONS, 'w') as f:
    all_pred_str = ','.join(all_pred_str)
    f.write(all_pred_str)




