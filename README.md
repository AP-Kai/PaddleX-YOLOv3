# PaddleX-YOLOv3目标检测一站式体验

十分钟跑通水果目标检测


##  一、项目背景

体验PaddleX-YOLOv3目标检测

##  二、数据集简介

数据集使用[水果目标检测数据集](https://aistudio.baidu.com/aistudio/datasetdetail/103740)，数据集已挂载到本项目环境下，运行下方指令即可解压。本数据集中共609张图片，包括苹果，橙子，香蕉。

### 1.数据集加载及划分


```python
! pip install paddlex
```


```python
! unzip data/data103740/fruit.zip -d data/
```


```python
import paddlex as pdx
import os
from tqdm import tqdm
from random import shuffle

dataset = 'data/fruit-detection'
train_txt = os.path.join(dataset, 'train.txt')
val_txt = os.path.join(dataset, 'val.txt')
lbl_txt = os.path.join(dataset, 'label_list.txt')

classes = ['apple', 'banana','orange']

with open(lbl_txt, 'w') as f:
    for l in classes:
        f.write(l+'\n')

xml_base = 'Annotations'
img_base = 'images'

xmls = [v for v in os.listdir(os.path.join(dataset, xml_base)) if v.endswith('.xml')]
shuffle(xmls)

split = int(0.9 * len(xmls))

with open(train_txt, 'w') as f:
    for x in tqdm(xmls[:split]):
        m = x[:-4]+'.jpg'
        xml_path = os.path.join(xml_base, x)
        img_path = os.path.join(img_base, m)
        f.write('{} {}\n'.format(img_path, xml_path))
    
with open(val_txt, 'w') as f:
    for x in tqdm(xmls[split:]):
        m = x[:-4]+'.jpg'
        xml_path = os.path.join(xml_base, x)
        img_path = os.path.join(img_base, m)
        f.write('{} {}\n'.format(img_path, xml_path))
```

### 2.数据集查看


```python
import numpy as np
import os

image_path = 'data/fruit-detection/images'
imgs = os.listdir(image_path)
infer_imgs = np.random.choice(imgs, 10)
infer_imgs
```


```python
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

%matplotlib inline
plt.figure(figsize=(16, 40))
for i in range(len(infer_imgs)):
    img = mpimg.imread('data/fruit-detection/images/'+infer_imgs[i])
    plt.subplot(5, 2, i+1)
    plt.imshow(img)
plt.show()
```

## 三、模型选择及开发

本项目选用了经典的网络MobileNetV1

### 配置GPU


```python
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
```

### 定义图像处理流程transforms


```python
# 数据增强
from paddlex.det import transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])
```

### 定义数据集Dataset


```python
train_dataset = pdx.datasets.VOCDetection(
    data_dir='data/fruit-detection',
    file_list='data/fruit-detection/train.txt',
    label_list='data/fruit-detection/label_list.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='data/fruit-detection',
    file_list='data/fruit-detection/val.txt',
    label_list='data/fruit-detection/label_list.txt',
    transforms=eval_transforms)
```

### 开始训练


```python
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV1')
model.train(
    num_epochs=280,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    learning_rate=0.00125,
    lr_decay_epochs=[210, 240],
    save_interval_epochs=20,
    save_dir='output/MobileNetV1',
    use_vdl=True)
```

### 模型预测


```python
import paddlex as pdx
model = pdx.load_model('output/MobileNetV1/best_model')
image_name = 'mixed_24.jpg'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.1, save_dir='./output/MobileNetV1')
```

## 四、效果展示

本项目可十分钟跑通全流程，如在本地运行，请更换相对应的路径再运行。

## 五、总结与升华

部分参考链接：

PaddleX快速上手-YOLOv3目标检测:[https://aistudio.baidu.com/aistudio/projectdetail/442375?channelType=0&channel=0](https://aistudio.baidu.com/aistudio/projectdetail/442375?channelType=0&channel=0)

PaddleX2.0快速上手-YOLOv3目标检测:[https://aistudio.baidu.com/aistudio/projectdetail/2160238?channelType=0&channel=0](https://aistudio.baidu.com/aistudio/projectdetail/2160238?channelType=0&channel=0)

## 个人简介

> AP-Kai 

> 沈阳工业大学 大一在读

> AI Studio: [https://aistudio.baidu.com/aistudio/personalcenter/thirdview/675310](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/675310)

> GitHub: [https://github.com/AP-Kai/AP-Kai](https://github.com/AP-Kai/AP-Kai)

> 本项目 AI Studio 链接：https://aistudio.baidu.com/aistudio/projectdetail/2285630?shared=1
