import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_augs = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

data_images = ImageFolder(root='../DataSet_Learn/classify-leaves', transform=train_augs)
train_csv=pd.read_csv('../DataSet_Learn/classify-leaves/train.csv')
class_to_num=train_csv.label.unique()
train_csv['class_num'] = train_csv['label'].apply(lambda x: np.where(class_to_num == x)[0][0])

class leaf_data(Dataset):
    def __init__(self,imgs,labels):
        self.imgs=imgs
        self.labels=labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        data = self.imgs[index][0]
        return data, label

imgs=data_images
labels=train_csv.class_num
leaf_data=leaf_data(imgs=imgs,labels=labels)
train_iter=DataLoader(dataset=leaf_data,batch_size=256,shuffle=True)
X, y = next(iter(train_iter))
X[0].shape, y[0]
(torch.Size([3, 224, 224]), torch.tensor(10))
# 展示一下
toshow = [torch.transpose(X[i],0,2) for i in range(16)]

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

show_images(toshow, 2, 8, scale=2)
