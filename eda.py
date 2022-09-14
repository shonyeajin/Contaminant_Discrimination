import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

label_path='./train/labels'
img_path='./train/images'
save_y='./crop_y/'
save_pred='./crop_pred/'
save_label='./crop_label/'

with open('./yolov5_2.json') as f:
  data=json.load(f)

err_flag=0
images=[]
labels=[]
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
  print(img_pth)
  image1=Image.open(img_pth)
  img_arr=np.array(image1)
  images.append(img_arr)
  labels.append(gt_list[0])

  print(gt_list)
  print(v)
  
temp=np.array(images)
temp_label=np.array(labels)
print(temp.shape)
print(temp_label.shape)
print(labels.count(0))
print(labels.count(1))
print(labels.count(2))


#### count plot ####
plt.figure(figsize=[10,5])
ax=sns.countplot(temp_label.ravel())
for p in ax.patches:
  height=p.get_height()
  ax.text(p.get_x()+p.get_width()/2., height+3, height, ha='center', size=15)
ax.set_ylim(0,350)
plt.show()


#### histogram analysis of channels ####
def histogram(arr):
    hists_r=[]
    hists_g=[]
    hists_b=[]
    color=('b','g','r')
    for j in range(len(arr)):
        for i, col in enumerate(color):
            hist=cv2.calcHist(arr[j], [i], None, [256], [0, 256])
            if i==0:
                hists_b.append(hist)
            elif i==1:
                hists_g.append(hist)
            elif i==2:
                hists_r.append(hist)
        

    hist_r=sum(hists_r)/len(hists_r)
    hist_g=sum(hists_g)/len(hists_g)
    hist_b=sum(hists_b)/len(hists_b)
    plt.plot(hist_b, color='b', label='Blue channel')
    plt.plot(hist_g, color='g',label='Green channel')
    plt.plot(hist_r, color='r',label='Red channel')
    plt.xlim([0,256])
    plt.ylim([0,50])
    plt.xlabel('Value of pixel')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

class_0=[]
for i in range(len(labels)):
  if labels[i]==0:
    class_0.append(images[i])
print(len(class_0))
histogram(class_0)

class_1=[]
for i in range(len(labels)):
  if labels[i]==1:
    class_1.append(images[i])
print(len(class_1))
histogram(class_1)

class_2=[]
for i in range(len(labels)):
  if labels[i]==2:
    class_2.append(images[i])
print(len(class_2))
histogram(class_2)