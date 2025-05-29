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

writer = SummaryWriter(log_dir= "logs/vit_experiment")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_helpers import train


transform = transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                               ])

train_data = datasets.CIFAR10(root="data",
                               train=True,
                               download = True,
                               transform = transform )
test_data = datasets.CIFAR10(root= 'data',
                              train = False,
                              download = True,
                              transform = transform)

## DataLoader
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data,batch_size=32,shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_vit = timm.create_model('mobilevit_xxs',pretrained = True,num_classes = 10)
model_vit.to(device)

dummy_input = torch.randn(1,3,224,224).to(device)
writer.add_graph(model_vit,dummy_input) # for visualizing model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_vit.parameters(),lr=0.001)
results = train(model_vit,train_loader,test_loader,optimizer,criterion,writer = writer)

torch.save(model_vit.state_dict(),"checkpoints/model_vit.pth")










