import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Model(nn.Module):
    def __init__(self, height, width):
        # super().__init__()
        super(Model, self).__init__()
        self.height = height
        self.width = width
        self.linear1 = nn.Linear(width, 128)       
        self.bn1 = nn.BatchNorm1d(128)          
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)      
        self.dropout1 = nn.Dropout(p=0.3)       
        self.leaky_relu = nn.LeakyReLU(0.1)     
        self.linear3 = nn.Linear(256, 128)      
        self.tanh = nn.Tanh()                   
        self.linear4 = nn.Linear(128, 64)       
        self.sigmoid = nn.Sigmoid()             
 
        self.linear5 = nn.Linear(64, 5)      
 
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.dropout1(x)
        x = self.leaky_relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        x = self.linear5(x)
        x = F.softmax(x, dim=-1)
        return x
    
# 手动添加
height = 4
width = 10
def get_inputs():
    return [torch.rand(height, width)]

def get_init_inputs():
    return [height, width]
