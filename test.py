#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from torchsummary import summary

from nets.unet import Unet

if __name__ == "__main__":
    model = Unet(num_classes=2).train().cuda()
    summary(model,(3,512,512))