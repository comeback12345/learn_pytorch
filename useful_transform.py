from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer=SummaryWriter("logs")
img=Image.open('classify-leaves/images/000005.jpg')
print(img)

tran_totensor=transforms.ToTensor()
img_tensor=tran_totensor(img)
writer.add_image("ToTensor",img_tensor,1)

#Normalize使用方法

print(img_tensor[0][0][0])
#print(img_tensor)
trans_norm=transforms.Normalize([0.5,0.2,0.3],[0.3,0.1,0.2])
img_norm=trans_norm(img_tensor)
writer.add_image("Normalize",img_norm,1)

#Resize使用方法

print(img.size)
trans_resize=transforms.Resize((40,40))#transform的Resize函数
#img PIL ->resize ->img_resize PIL
img_resize=trans_resize(img)
#img_resize PIL ->totensor ->img_resize tensor
img_resize=tran_totensor(img_resize)
trans_resize_2=transforms.Resize(10)
#PIL ->PIL ->tensor
trans_compose=transforms.Compose([trans_resize_2,tran_totensor])#transform的compose函数
img_resize_2=trans_compose(img)#compose函数生成的是tensor类型的数据
writer.add_image("Resize",img_resize_2,1)

#RandomCrop

teans_random=transforms.RandomCrop(30)
trans_compose_2=transforms.Compose([teans_random,tran_totensor])
print(type())
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()