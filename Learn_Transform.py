from PIL import Image
from torchvision import transforms

img_path='hymenoptera_data/train/ants/0013035.jpg'
img=Image.open(img_path)

print(type(img))
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)#将jpg的文件改成为tensor文件

print(type(tensor_img))