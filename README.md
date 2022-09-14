# Contaminant_Discrimination ⏰


## Introduction
  **Region-of-Interest based Histogram Similarity Comparison for Contaminant Discrimination** source code archive
  
## Installation & Data preparation
  - Clone this repository
  
  - Organize them as following:
    ```
    Contaminant_Discrimination/
      ├── crop_label/
      ├── crop_y/
      ├── crop_pred/
      ├── train/
      ├── test/
      ├── valid/
      └── yolov5/
    ```
    
  - Download dataset from
      - [train](https://drive.google.com/drive/folders/1oLK0JogbWu88Z_olPe67mpgxE-3zJ-0P?usp=sharing)
      - [valid](https://drive.google.com/drive/folders/1l7D8u5SRGEAGl3q3Ta2jj8FkWO9zAbkv?usp=sharing)
      - [test](https://drive.google.com/drive/folders/1pOw5VBteoUpg7_ua9X0GjFnk9u_87-C8?usp=sharing)
      - [crop_label](https://drive.google.com/drive/folders/1NZgo54a1FrdFnjT3VrsbV3lVlYz7U2LY?usp=sharing)
      - [crop_y](https://drive.google.com/drive/folders/1P2Lh0Lh-UYdSgCo2uYa9IFcqvH9gltvr?usp=sharing)
      - [crop_pred](https://drive.google.com/drive/folders/1-24mYdQrIwCVmSqFormeGv-ysY1pDw-O?usp=sharing)
   
   - Clone object detection architectures and models
     - [yolov5](https://github.com/ultralytics/yolov5)



## Run
 1. How to train
    - data.yaml 'train', 'val', 'test' 경로 수정하기
    - cd yolov5
    - python train.py --img 416 --batch 16 --epoch 50 --data ../data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name [결과 파일 이름]

 2. 크롭 이미지 생성하기 -> crop.py 실행
 3. 유사도 비교하기 -> similarity.py 실행
 4. 데이터 분포 확인하기 -> eda.py 실행
  

## 제안 방법
![제안방법](https://user-images.githubusercontent.com/55689863/189947587-5b2276d5-a5a6-4361-b601-7aed400c2032.png)

## 개발환경
colab
