## Unet：U-Net: Convolutional Networks for Biomedical Image Segmentation目标检测模型在Pytorch当中的实现
---

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [miou计算 miou](#miou计算)
8. [参考资料 Reference](#Reference)

## 性能情况
**unet并不适合VOC此类数据集，其更适合特征少，需要浅层特征的医药数据集之类的。**
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [unet_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_voc.pth) | VOC-Val12 | 512x512| 55.11 | 

## 所需环境
torch==1.2.0    
torchvision==0.4.0   

## 注意事项
unet_voc.pth是基于VOC拓展数据集训练的。  
unet_medical.pth是使用示例的细胞分割数据集训练的。  
在使用时需要注意区分。  

## 文件下载
训练所需的unet_voc.pth和unet_medical.pth可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1b-mQ7DjialleyagmuUWYMQ 提取码: 97pw    

VOC拓展数据集的百度网盘如下：  
链接: https://pan.baidu.com/s/1BrR7AUM1XJvPWjKMIy2uEw 提取码: vszf     

## 预测步骤
### 一、使用预训练权重
#### a、VOC预训练权重
1. 下载完库后解压，如果想要利用voc训练好的权重进行预测，在百度网盘或者release下载unet_voc.pth，放入model_data，运行即可预测。  
```python
img/street.jpg
```    
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。      
#### b、医药预训练权重
1. 下载完库后解压，如果想要利用医药数据集训练好的权重进行预测，在百度网盘或者release下载unet_medical.pth，放入model_data，修改unet.py中的model_path和num_classes；
```python
_defaults = {
    "model_path"        : 'model_data/unet_voc.pth',
    "model_image_size"  : (512, 512, 3),
    "num_classes"       : 21,
    "cuda"              : True,
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"             : True
}

```
2. 运行即可预测。  
```python
img/cell.png
```
### 二、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在unet.py文件里面，在如下部分修改model_path、backbone和num_classes使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    "model_path"        : 'model_data/unet_voc.pth',
    "model_image_size"  : (512, 512, 3),
    "num_classes"       : 21,
    "cuda"              : True,
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"             : True
}
```
3. 运行predict.py，输入    
```python
img/street.jpg
```   
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

## 训练步骤
### 一、训练voc数据集
1. 将我提供的voc数据集放入VOCdevkit中（无需运行voc2unet.py）。  
2. 在train.py中设置对应参数，默认参数已经对应voc数据集所需要的参数了，所以只要修改backbone和model_path即可。  
3. 运行train.py进行训练。  

### 二、训练自己的数据集
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4. 在训练前利用voc2unet.py文件生成对应的txt。
5. 注意修改train.py的num_classes为分类个数+1。  
6. 运行train.py即可开始训练。  

### 三、训练医药数据集
1. 下载VGG的预训练权重到model_data下面。  
2. 按照默认参数运行train_medical.py即可开始训练。

## miou计算
参考miou计算视频和博客。  

## Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus
