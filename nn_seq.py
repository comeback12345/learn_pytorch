import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui,self).__init__()
#         from torch.nn import Conv2d
#         self.conv1=Conv2d(3,32,5,padding=2)
#         self.maxpool1=MaxPool2d(2)
#         self.conv2 = Conv2d(32, 32, 5, padding=2)
#         self.maxpool2 = MaxPool2d(2)
#         self.conv3 = Conv2d(32, 64, 5, padding=2)
#         self.maxpool3 = MaxPool2d(2)
#         self.flatten=Flatten()#64*4*4
#         self.linear1=Linear(1024,64)
#         self.linear2=Linear(64,10)
#
#     def forward(self,x):
#         x=self.conv1(x)
#         x=self.maxpool1(x)
#         x=self.conv2(x)
#         x=self.maxpool2(x)
#         x = self.conv3(x)
#         x = self.maxpool3(x)
#         x=self.flatten(x)
#         x=self.linear1(x)
#         x=self.linear2(x)
#         return x

#另外的一种写法，使用Sequential，代码会更加的简洁
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


#以下是实验性质的测试
tudui=Tudui()
input=torch.ones((64,3,32,32))
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

writer=SummaryWriter("logs")
writer.add_graph(tudui,input)
writer.close()