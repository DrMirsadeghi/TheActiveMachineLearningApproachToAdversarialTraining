import os, glob
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import mair
from mair.attacks import PGD
from mair.defenses.advtraining.advtrainer import AdvTrainer
from mair.defenses.advtraining.standard import Standard


FROZEN_MODEL_PATH= F"fm10.pt"

PATH = "./models/"
NAME = "Sample"
SAVE_PATH = PATH + NAME
MODEL_NAME = "ResNet18"
DATA = "CIFAR10"
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
N_VALIDATION = 1000
N_CLASSES = 10
EPOCH = 200
EPS = 8/255
ALPHA = 2/255
STEPS = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[4].split('_')[0]]
        #label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

#transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))

import sys
setting= int(sys.argv[1])
if setting >=0 and setting<=2:
    print("You Selected {}".format(setting))
    print("0) WideResNet-34-10 for CIFAR-10")
    print("1) WideResNet-34-10 for CIFAR-100")
    print("2) PreAct ResNet for SVHN")
else:
    print("Choice must be in range[0,2]")
    exit()
if setting==0:
    train_data = dsets.CIFAR10(root='./data',
                           train=True,
                           download=True,
                           transform=transform)
    test_data  = dsets.CIFAR10(root='./data',
                           train=False,
                           download=True,
                           transform=transform)
elif setting==1:
    train_data = dsets.CIFAR100(root='./data',
                           train=True,
                           download=True,
                           transform=transform)
    test_data  = dsets.CIFAR100(root='./data',
                           train=False,
                           download=True,
                           transform=transform)
    '''
    id_dict = {}
    for i, line in enumerate(open('tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i

    transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
    train_data = TrainTinyImageNetDataset(id=id_dict, transform = transform)
    test_data = TestTinyImageNetDataset(id=id_dict, transform=transform)
    '''
elif setting==2:
    '''
    train_data = dsets.MNIST(root='./data',
                           train=True,
                           download=True,
                           transform=transform)
    test_data  = dsets.MNIST(root='./data',
                           train=False,
                           download=True,
                           transform=transform)
    '''
    train_data = dsets.SVHN(root='./data',
                           split='train',
                           download=True,
                           transform=transform)
    test_data  = dsets.SVHN(root='./data',
                           split='test',
                           download=True,
                           transform=transform)

batch_size = 128
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=False)
if setting==0:
    MODEL_NAME='ResNet18'
elif setting==1:
    MODEL_NAME='WRN34-10'
    N_CLASSES=200
elif setting==2:
    MODEL_NAME='ResNet18'


model_choice=int(sys.argv[2])

if model_choice==0:
    MODEL_NAME='ResNet18'
elif model_choice==1:
    MODEL_NAME='WRN34-10'

model = mair.utils.load_model(model_name=MODEL_NAME,
                              n_classes=N_CLASSES).cuda() # Load model
rmodel = mair.RobModel(model, n_classes=N_CLASSES,
                       normalization_used={'mean':MEAN, 'std':STD}).cuda()
trainer = Standard(rmodel)
trainer.record_rob(train_loader, test_loader, eps=EPS, alpha=2/255, steps=10, std=0.1,
                   n_train_limit=N_VALIDATION, n_val_limit=N_VALIDATION)
trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9, weight_decay=0.0005)",
              scheduler="Step(milestones=[100, 150], gamma=0.1)",
              scheduler_type="Epoch",
              minimizer=None,
              n_epochs=10, n_iters=len(train_loader)
             )
trainer.fit(train_loader=train_loader, n_epochs=10,
            save_path=SAVE_PATH, save_best={"Clean(Val)":"HBO", "PGD(Val)":"HB"},
            save_type=None, save_overwrite=True, record_type="Epoch")
torch.save(rmodel.state_dict(), FROZEN_MODEL_PATH)
