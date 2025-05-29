import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import timm
from tqdm import tqdm
import sys
import os
from torch.utils.tensorboard import SummaryWriter
import yaml
with open("config.yml",'r') as f:
    config = yaml.safe_load(f)

train_config = config["train"]
model_config = config["model"]
dataset_config = config["dataset"]
device = config["device"]


writer = SummaryWriter(log_dir= "logs/vit_experiment")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_helpers import train


transform = transforms.Compose([transforms.Resize((dataset_config['image_size'],dataset_config['image_size'])),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=dataset_config['normalize_mean'], std=dataset_config['normalize_std'])
                               ])

train_data = datasets.CIFAR10(root=dataset_config['data_dir'],
                               train=True,
                               download = True,
                               transform = transform )
test_data = datasets.CIFAR10(root= dataset_config['data_dir'],
                              train = False,
                              download = True,
                              transform = transform)

## DataLoader
train_loader = DataLoader(train_data,batch_size=train_config['batch_size'],shuffle=True)
test_loader = DataLoader(test_data,batch_size=train_config['batch_size'],shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_vit = timm.create_model(model_config['name'],pretrained = model_config["pretrained"],num_classes = 10)
model_vit.to(device)

dummy_input = torch.randn(1,3,224,224).to(device)
writer.add_graph(model_vit,dummy_input) # for visualizing model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_vit.parameters(),lr=train_config["learning_rate"])
results = train(model_vit,train_loader,test_loader,optimizer,criterion,writer = writer,epochs=train_config['epochs'])

torch.save(model_vit.state_dict(),"checkpoints/model_vit.pth")










