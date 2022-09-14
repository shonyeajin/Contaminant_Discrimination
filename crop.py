import json
import sys
from PIL import Image
import os
import cv2 as cv

with open('./yolov5_2.json') as f:
  data=json.load(f)
  
label_path='/content/drive/MyDrive/Colab Notebooks/vision_project/cleaning/train/labels'
img_path='/content/drive/MyDrive/Colab Notebooks/vision_project/cleaning/train/images'
save_y='/content/drive/MyDrive/Colab Notebooks/vision_project/cleaning/crop_y/'
save_pred='/content/drive/MyDrive/Colab Notebooks/vision_project/cleaning/crop_pred/'
save_label='/content/drive/MyDrive/Colab Notebooks/vision_project/cleaning/crop_label/'

err_flag=0
for k in range(len(data['result'])):
  s=data['result'][k]['title']
  v=data['result'][k]['point']
  index=s.rfind('/')
  name=label_path+s[index:-3]+'txt'
  f=open(name, 'r')
  gt=f.read()
  f.close()
  gt=gt.split('\n')
  gt_list=[]
  for i in range(len(gt)):
    gt_list.append(gt[i].split(' '))

  for i in range(len(gt_list)):
    for j in range(len(gt_list[i])):
      try:
        gt_list[i][j]=float(gt_list[i][j])
      except:
        err_flag=1
        break
    if err_flag==1:
      break

  if err_flag==1:
    err_flag=0
    continue

  temp=[]
  for i in range(len(gt_list)):
    temp.append(abs(gt_list[i][1]-v[0]))
  minidx=temp.index(min(temp))
  gt_list=gt_list[minidx]

  img_pth=img_path+s[index:]
  image1=Image.open(img_pth)
  gt_list=list(map(lambda x: x*image1.size[0], gt_list))
  v=list(map(lambda x: x*image1.size[0], v))

  x=gt_list[1]
  y=gt_list[2]
  w=gt_list[3]
  h=gt_list[4]

  crop_p=image1.crop((x-w/2, y-h/2, x+w/2, y+h/2))

  x=v[0]
  y=v[1]
  w=v[2]
  h=v[3]

  crop_g=image1.crop((x-w/2, y-h/2, x+w/2, y+h/2))
  crop_p.save(save_pred+str(k)+'.jpg')
  crop_g.save(save_y+str(k)+'.jpg')

  f=open(save_label+str(k)+'.txt','w')
  f.write(str(gt_list[0]))
  f.close()
