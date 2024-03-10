import os
path = '../DataSet_Learn/classify-leaves/images'
file_list = os.listdir(path)
for file in file_list:
    front,end=file.split('.')
    front=front.zfill(6)
    new_name='.'.join([front,end])
    os.rename(path+'/'+file,path+'/'+new_name)#改名的代码
# for file in file_list:
#     front,end=file.split('.')
#     new_front= front[:5]#只获取后面5位的名称
#     new_name='.'.join([new_front,end])
#     os.rename(path+'/'+file,path+'/'+new_name)
file_list = os.listdir(path)
print(file_list)
