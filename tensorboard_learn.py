from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer=SummaryWriter("logs")
image_path='hymenoptera_data/train/bees/16838648_415acd9e3f.jpg'
img=Image.open(image_path)
img_array=np.array(img)
img.show()
print(img_array.shape)
writer.add_image("test",img_array,2,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close