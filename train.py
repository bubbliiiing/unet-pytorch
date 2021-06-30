import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.unet import Unet
from nets.unet_training import CE_Loss, Dice_loss, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.metrics import f_score


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    net = net.train()
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size: 
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(imgs)
            loss    = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs  = net(imgs)
                val_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    val_loss  = val_loss + main_dice
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()
                
            
            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    loss_history.append_loss(total_loss/(epoch_size+1), val_toal_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))


if __name__ == "__main__":
    log_dir = "logs/"   
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    inputs_size = [512,512,3]
    #---------------------#
    #   分类个数+1
    #   2+1
    #---------------------#
    NUM_CLASSES = 21
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = False
    #-------------------------------#
    #   主干网络预训练权重的使用
    #-------------------------------#
    pretrained = False
    #-------------------------------#
    #   Cuda的使用
    #-------------------------------#
    Cuda = True
    #------------------------------#
    #   数据集路径
    #------------------------------#
    dataset_path = "VOCdevkit/VOC2007/"

    model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train()
    
    loss_history = LossHistory("logs/")
    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    model_path = r"model_data/unet_voc.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-4
        Init_Epoch      = 0
        Interval_Epoch  = 50
        Batch_size      = 2
        
        optimizer       = optim.Adam(model.parameters(),lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

        train_dataset   = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True, dataset_path)
        val_dataset     = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False, dataset_path)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for param in model.vgg.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Interval_Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Interval_Epoch,Cuda)
            lr_scheduler.step()
    
    if True:
        lr              = 1e-5
        Interval_Epoch  = 50
        Epoch           = 100
        Batch_size      = 2

        optimizer       = optim.Adam(model.parameters(),lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

        train_dataset   = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True, dataset_path)
        val_dataset     = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False, dataset_path)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for param in model.vgg.parameters():
            param.requires_grad = True

        for epoch in range(Interval_Epoch,Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda)
            lr_scheduler.step()

