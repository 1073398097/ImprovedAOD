import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import numpy as np
from torchvision import transforms
import datetime
from net import dehaze_net

import torch.nn.functional as F
from ssim import ssim as sm

import matplotlib.pyplot as plt


def composite_loss(clean_image, img_orig):
    # 图像重构损失
    recon_loss = F.mse_loss(clean_image, img_orig)
    # import pdb
    # pdb.set_trace()

    # 图像结构相似损失
    # #ssim = ssim_value = SSIM(img1, img2, multichannel=True)
    # clean_image_np = clean_image.detach().cpu( ).numpy( ).copy( )
    # img_orig_np = img_orig.detach().cpu( ).numpy( ).copy( )
    # clean_image_np_3d = clean_image_np[:, -3:, :, :]
    # img_orig_np_3d = img_orig_np[:, -3:, :, :]

    # import pdb;pdb.set_trace()

    ssim=sm(clean_image, img_orig)
    ssim_loss = 1 - ssim

    # TV损失
    tv_loss_x = torch.mean(torch.abs(clean_image[:, :, :, :-1] - clean_image[:, :, :, 1:]))
    tv_loss_y = torch.mean(torch.abs(clean_image[:, :, :-1, :] - clean_image[:, :, 1:, :]))
    tv_loss = tv_loss_x + tv_loss_y

    # 组合损失
    composite_loss = 1 * recon_loss + 0.84* ssim_loss + 2e-8 * tv_loss +9e-5

    return composite_loss

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


def weights_init(m):  # 初始化权重
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(config):

    #去雾网络实例化
    net1 = dehaze_net(in_channels=3)
    net1.cuda( )
    net2 = dehaze_net(in_channels=6)
    net2.cuda( )
    net3 = dehaze_net(in_channels=6)
    net3.cuda( )

    net1.apply(weights_init)
    net2.apply(weights_init)
    net3.apply(weights_init)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path)

    val_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="val")



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False,
                                               num_workers=config.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

    #criterion = nn.MSELoss( ).cuda( )   #均方误差损失函数

    #criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(net1.parameters( ), lr=config.lr, weight_decay=config.weight_decay)
    optimizer1 = torch.optim.Adam(net2.parameters( ), lr=config.lr, weight_decay=config.weight_decay)
    optimizer2 = torch.optim.Adam(net3.parameters( ), lr=config.lr, weight_decay=config.weight_decay)

    net1.train( )
    net2.train( )
    net3.train( )

    train_loss = []  # 记录损失函数

    alltime = 0.0
    for epoch in range(config.num_epochs):
        start_time = time.time( )  # 记录开始时间

        for iteration, (img_orig, img_haze) in enumerate(train_loader): #No such file or directory: 'E:\\wangchao\\mist_dataset\\RESIDE\\OTSBETA\\allhazy\\2121_0.85_0.2.jpg'

            img_orig = img_orig.cuda( )
            img_haze = img_haze.cuda( )  # all 8 3 640 480

            # 下采样
            img_haze_2x = nn.MaxPool2d(2)(img_haze)  # 8 3 320 240
            img_haze_4x = nn.MaxPool2d(4)(img_haze)  # 8 3 160 120

            # Scale3
            img_haze_4x = img_haze_4x.cuda( )
            J_4x = net1(img_haze_4x,img_haze_4x)  # 8 3 160 120
            upsample = nn.Upsample(scale_factor=2)
            J_4x_up = upsample(J_4x)  # 8  3 320 240

            npimg = J_4x_up.numpy( )  # plt输入需要时ndarray
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')  # 需要将通道转到最后一维
            plt.show( )

            import pdb
            pdb.set_trace()

            # Scale2
            img_haze_2x = img_haze_2x.cuda( )
            J_4x_up = J_4x_up.cuda( )
            img_concat_2x = torch.cat((J_4x_up, img_haze_2x), dim=1)
            J_2x = net2(img_concat_2x,img_haze_2x)
            J_2x_up = upsample(J_2x)

            # Scale1
            img_concat_1x = torch.cat((img_haze, J_2x_up), dim=1)
            clean_image = net3(img_concat_1x,img_haze)


            #loss = criterion(clean_image, img_orig)

            loss = composite_loss(clean_image, img_orig)  # 损失样例

            import pdb
            pdb.set_trace()

            train_loss.append(loss.item( ))  # 损失加入到列表中

            optimizer.zero_grad( )
            optimizer1.zero_grad( )
            optimizer2.zero_grad( )

            loss.backward( )

            torch.nn.utils.clip_grad_norm_(net1.parameters( ), config.grad_clip_norm) # 防止梯度爆炸(gradient explosion)的问题
            torch.nn.utils.clip_grad_norm_(net2.parameters( ), config.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(net3.parameters( ), config.grad_clip_norm)

            optimizer.step( )
            optimizer1.step( )
            optimizer2.step( )

            if ((iteration + 1) % config.display_iter) == 0:
                print("Epoch[",epoch+1,"/",config.num_epochs,"]-Loss at iteration", iteration + 1, ":", loss.item( ))
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(net3.state_dict( ), config.snapshots_folder + "Scale1/Epoch" + str(epoch) + '.pth')
                torch.save(net2.state_dict( ), config.snapshots_folder + "Scale2/Epoch" + str(epoch) + '.pth')
                torch.save(net1.state_dict( ), config.snapshots_folder + "Scale3/Epoch" + str(epoch) + '.pth')

        current_date = datetime.date.today( )
        formatted_date = current_date.strftime("%m_%d_")
        # 定义文件名
        loss_shuju = f"{formatted_date}loss.txt"
        with open("loss/" + loss_shuju, 'w') as train_los:
            train_los.write(str(train_loss))

        end_time = time.time( )  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算执行时间
        alltime= alltime+elapsed_time
        print('Epoch',epoch+1,'训练时间:',int(elapsed_time//60),'min',elapsed_time-int(elapsed_time//60)*60, 's')
        print('目前总耗时：',int(alltime//60//60),'h',int(alltime // 60 - (alltime // 60 // 60)*60),'min',int(alltime-(alltime//60)*60),'s')

        # Validation Stage
        start_time = time.time( )  # 记录开始时间
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):

            img_orig = img_orig.cuda( )
            img_haze = img_haze.cuda( )

            # 下采样
            img_haze_2x = nn.MaxPool2d(2)(img_haze)  # 8 3 320 240
            img_haze_4x = nn.MaxPool2d(4)(img_haze)  # 8 3 160 120

            # Scale3
            img_haze_4x = img_haze_4x.cuda( )
            J_4x = net1(img_haze_4x,img_haze_4x)  # 8 3 160 120
            upsample = nn.Upsample(scale_factor=2)
            J_4x_up = upsample(J_4x)  # 8  3 320 240

            # Scale2
            img_haze_2x = img_haze_2x.cuda( )
            J_4x_up = J_4x_up.cuda( )
            img_concat_2x = torch.cat((J_4x_up, img_haze_2x), dim=1)
            J_2x = net2(img_concat_2x,img_haze_2x)
            J_2x_up = upsample(J_2x)

            # Scale1
            img_concat_1x = torch.cat((img_haze, J_2x_up), dim=1)
            clean_image = net3(img_concat_1x,img_haze)
            # import pdb
            # pdb.set_trace()
            if ((iter_val + 1) % 5) == 0:
                print("now at iter_val", iter_val + 1)

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         config.sample_output_folder + str(iter_val + 1) + ".jpg")

        torch.save(net3.state_dict( ), config.snapshots_folder + "dehazer.pth")
        end_time = time.time( )  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算执行时间
        alltime = alltime + elapsed_time
        print('Epoch', epoch + 1, '验证使用的时间:', int(elapsed_time // 60), 'min',int(elapsed_time - (elapsed_time // 60) * 60), 's')
        print('目前总耗时：', int(alltime // 60 // 60), 'h', int(alltime // 60 - (alltime // 60 // 60)*60), 'min',
              int(alltime - (alltime // 60) * 60), 's')


if __name__ == "__main__":

    parser = argparse.ArgumentParser( )

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="data/images/")
    parser.add_argument('--hazy_images_path', type=str, default="data/data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")

    config = parser.parse_args( )

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)
