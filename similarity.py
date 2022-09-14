import json
import sys
from PIL import Image
import os
import cv2 as cv
import cv2
import numpy as np
import matplotlib.pylab as plt
from numpy import dot
from numpy.linalg import norm

label_path='./train/labels'
img_path='./train/images'
save_y='./crop_y/'
save_pred='./crop_pred/'
save_label='./crop_label/'

def cos_sim(A, B):
  return dot(A,B)/(norm(A)*norm(B))

with open('./yolov5_2.json') as f:
  data=json.load(f)

res1=[]
res2=[]
res3=[]
res4=[]
res5=[]

res1_0=[]
res2_0=[]
res3_0=[]
res4_0=[]
res5_0=[]

res1_1=[]
res2_1=[]
res3_1=[]
res4_1=[]
res5_1=[]

res1_2=[]
res2_2=[]
res3_2=[]
res4_2=[]
res5_2=[]

for k in os.listdir('./crop_y/'):
  if k[0]=='.':
    continue
  p_path=save_pred+k
  g_path=save_y+k
  l_path=save_label+k
  imgs = []
  imgs.append(cv2.imread(g_path))
  imgs.append(cv2.imread(p_path))
  f=open(save_label+k[:-3]+'txt', 'r')
  gt=f.read()
  f.close()
  gt=int(float(gt))
  gt=gt/416

  hists = []
  for img in imgs:
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      hist = cv2.calcHist([hsv], [0], None, [256], [0,256])
      cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
      hists.append(hist)

  query = hists[0]

  costemp1=[]
  costemp2=[]
  for i in range(len(hists[0])):
    costemp1.append(hists[0][i][0])
  for i in range(len(hists[1])):
    costemp2.append(hists[1][i][0])
  cosval=cos_sim(costemp1, costemp2)
  res5.append(cosval)
  if gt==0.0:
    res5_0.append(cosval)
  elif gt==1.0:
    res5_1.append(cosval)
  elif gt==2.0:
    res5_2.append(cosval)

  methods = ['CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA', 'EMD']

  for index, name in enumerate(methods):
      print('%-10s' % name, end = '\t')  
      
      for i, histogram in enumerate(hists):
          ret = cv2.compareHist(query, histogram, index) 
          
          if index == cv2.HISTCMP_INTERSECT:                   
              ret = ret/np.sum(query)
          if i==1:                        
            if index==0:
              res1.append(ret)
              if gt==0.0:
                res1_0.append(ret)
              elif gt==1.0:
                res1_1.append(ret)
              elif gt==2.0:
                res1_2.append(ret)

            elif index==1:
              res2.append(ret)   
              if gt==0.0:
                res2_0.append(ret)
              elif gt==1.0:
                res2_1.append(ret)
              elif gt==2.0:
                res2_2.append(ret)

            elif index==2:
              res3.append(ret) 
              if gt==0.0:
                res3_0.append(ret)
              elif gt==1.0:
                res3_1.append(ret)
              elif gt==2.0:
                res3_2.append(ret)
    
            elif index==3:
              res4.append(ret)
              if gt==0.0:
                res4_0.append(ret)
              elif gt==1.0:
                res4_1.append(ret)
              elif gt==2.0:
                res4_2.append(ret)
                      
          print("img%d :%7.2f"% (i+1 , ret), end='\t')
      print()

print('res1', np.mean(res1))
print('res2', np.mean(res2))
print('res3', np.mean(res3))
print('res4', np.mean(res4))
print('res5', np.mean(res5))

print('res1_0', np.mean(res1_0))
print('res1_1', np.mean(res1_1))
print('res1_2', np.mean(res1_2))

print('res2_0', np.mean(res2_0))
print('res2_1', np.mean(res2_1))
print('res2_2', np.mean(res2_2))

print('res3_0', np.mean(res3_0))
print('res3_1', np.mean(res3_1))
print('res3_2', np.mean(res3_2))

print('res4_0', np.mean(res4_0))
print('res4_1', np.mean(res4_1))
print('res4_2', np.mean(res4_2))

print('res5_0', np.mean(res5_0))
print('res5_1', np.mean(res5_1))
print('res5_2', np.mean(res5_2))




res1=[]
res2=[]
res3=[]
res4=[]

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

  if gt_list[0]==0.0:
    res4.append(cos_sim(v, gt_list[1:]))
    res1.append(cos_sim(v, gt_list[1:]))
  if gt_list[0]==1.0:
    res4.append(cos_sim(v, gt_list[1:]))
    res2.append(cos_sim(v, gt_list[1:]))
  if gt_list[0]==2.0:
    res4.append(cos_sim(v, gt_list[1:]))
    res3.append(cos_sim(v, gt_list[1:]))

print('res1:',np.mean(res1))
print('res2:',np.mean(res2))
print('res3:',np.mean(res3))
print('res4:',np.mean(res4))