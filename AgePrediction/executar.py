import os
import sys
import time
import argparse
from utils.utils import *
from utils.datasets import *
from models.models import *


parser = argparse.ArgumentParser()
parser.add_argument('--cuda',
                    type=int,
                    default=-1)

parser.add_argument('--numworkers',
                    type=int,
                    default=3)

parser.add_argument('--seed',
                    type=int,
                    default=-1)

parser.add_argument('--outpath',
                    type=str,
                    required=True)

parser.add_argument('--imp_weight',
                    type=int,
                    default=0)

parser.add_argument('--dataset',
                    type=str,
                    default='CACD')

parser.add_argument('--loss',
                    type=str,
                    default='ce')

args = parser.parse_args()

NUM_WORKERS = args.numworkers
DATASET = args.dataset
LOSS = args.loss

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

IMP_WEIGHT = args.imp_weight

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')


path_list = return_paths(DATASET)
TRAIN_CSV_PATH = path_list[0]
VALID_CSV_PATH = path_list[1]
TEST_CSV_PATH = path_list[2]
IMAGE_PATH = path_list[3]


header = []
header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Output Path: %s' % PATH)
header.append('Script: %s' % sys.argv[0])

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()


##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate = 0.0005
num_epochs = 200

# Architecture
NUM_CLASSES = 26
BATCH_SIZE = 256
GRAYSCALE = False


# Nomes necessitem imp per a coral i ordinal
if(LOSS!='ce'):
    df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
    ages = df['age'].values
    del df
    ages = torch.tensor(ages, dtype=torch.float)
    imp = task_importance_weights(ages, IMP_WEIGHT)
else:
    imp = torch.zeros(NUM_CLASSES - 1, dtype=torch.float)
imp = imp.to(DEVICE)



# TRANSFORMACIONS (igual per a totes)
train_loader, valid_loader, test_loader = df_loader(TRAIN_CSV_PATH, VALID_CSV_PATH,
                                                    TEST_CSV_PATH, IMAGE_PATH, BATCH_SIZE, NUM_WORKERS)

# cost_fn ho cridarem mes endavant des de dins!

# CREEM MODEL I OPTIMITZADOR
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = resnet34(NUM_CLASSES, GRAYSCALE)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


start_time = time.time()
# COMENÇA LA PART DE CRIDA AL MODEL
best_mae, best_rmse, best_epoch = 999, 999, -1
for epoch in range(num_epochs):
    model.train()

    # FICARHO EN UNA FUNCIO SEPARAT???
    for batch_idx, () in enumerate(train_loader):
        s=0