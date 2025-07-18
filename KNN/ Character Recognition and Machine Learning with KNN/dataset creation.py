import os
import numpy as np

data_path='dataset'

categories=os.listdir(data_path)
print(categories)
labels=[i for i in range(len(categories))]
print(labels)
category_dict={'a':0,'ae':1,'e':2,'u':3}
print(category_dict)

import cv2

data=[]
target=[]

for category in categories:
    imgs_path=os.path.join(data_path,category)
    img_names=os.listdir(imgs_path)
    #print(imgs_path,img_names)
    print(category,'---------------------')
    for img_name in img_names:
        img_path=os.path.join(imgs_path,img_name)
        #print(img_path)
        img=cv2.imread(img_path,0)
        img=cv2.resize(img,(8,8))
        data.append(img)
        target.append(category_dict[category])
        #cv2.imshow('LIVE',img)
        #k=cv2.waitKey(100)
        #if(k==27):
            #break
#cv2.destroyAllWindows()


print(len(data),len(target))


from matplotlib import pyplot as plt

plt.imshow(data[10],cmap='gray')

data[10]

target[10]

data=np.array(data)
print('before resize:',data.shape)

target=np.array(target)
print('before resize:',target.shape)


data=np.array(data)
print('before resize:',data.shape)
data=data.reshape(data.shape[0],data.shape[1]*data.shape[2])
#data=data.reshape(270,8*8)
print('after resize:',data.shape)
target=np.array(target)

np.save('data',data)
np.save('target',target)

print(target.shape)
print(target)