
from time import time
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Metric
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


class FTModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=1e-3):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        
        # transfer learning if pretrained=True
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        # layers are frozen by using eval()
        self.feature_extractor.eval()
        # freeze params
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        n_sizes = self._get_conv_output(input_shape)
        
        self.classifier = nn.Linear(n_sizes,num_classes)

#         self.classifier = nn.Sequential(nn.Linear(n_sizes, 256),
#                                         nn.ReLU(),
#                                         nn.Linear(256, 128),
#                                         nn.ReLU(),
#                                         nn.Linear(128,num_classes),
#                                        )
        self.n_sizes = n_sizes
  
    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(tmp_input) 
        print("bottleneck output size",output_feat.shape)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x
    
    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_nb):
        x,y = batch
        yp = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(yp, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_nb, dataloader_idx):
        x,y = batch
        yp = self(x)
        return yp
    
    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)
        return adam

def run():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Resize(224),
         ])
    path = '/raid/data/ml'
    BS = 1024
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    x = trainset[0][0]
    x.min(), x.max(), x.shape

    dataiter = iter(trainloader)

    net = FTModel((3,224,224),10,2e-4)

    net.n_sizes

    net.summarize()

    epochs = 5
    trainer = pl.Trainer(gpus=4, max_epochs=epochs, accelerator='ddp',
                         progress_bar_refresh_rate=20,precision=16)

    start = time()
    trainer.fit(net, trainloader)
    duration = time() - start
    print(f"Training time: {duration:.2f} Seconds")

    test_x_ds = X_only_DS(testset)
    test_x_loader = torch.utils.data.DataLoader(test_x_ds, batch_size=256,
                                                shuffle=False, num_workers=8)
    yp = trainer.predict(net,test_x_loader)
    # -

    yp = torch.cat(yp,dim=0)
    yp.shape

    ytest = [y for _,y in testloader]
    ytest = torch.cat(ytest,dim=0)
    ytest.shape

    print("Accuracy",(yp.argmax(dim=1) == ytest.cuda()).float().mean())

if __name__ == '__main__':
    run()
    

# ### Training with DP is neither faster nor better. 
# - 1 GPU: Training time 3 min 38s Accuracy 77%
# - DP 4 GPUs: Training time 3min 52s Accuracy 76%
