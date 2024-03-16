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
import net
import numpy as np
from torchvision import transforms
from PIL import Image,ImageDraw, ImageFont
import glob
import cv2
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import matplotlib.pyplot as plt
import pandas as pd
# RRDNet
from RRDNet.RRDNet import RRDNet
from RRDNet.loss_functions import reconstruction_loss, illumination_smooth_loss, reflectance_smooth_loss, noise_loss, normalize01
import RRDNet.config as config
import torch.optim as optim


import csv
import datetime

def pipline_retinex(net, img):
    #img_tensor = transforms.ToTensor()(img)  # [b,c,h,w]
    img_tensor = img  # [b,c,h,w]
    img_tensor = img_tensor.to(config.device)
    #img_tensor = img_tensor.unsqueeze(0)  # [1, c, h, w]

    optimizer = optim.Adam(net.parameters( ), lr=config.lr)

    # iterations
    for i in range(config.iterations + 1):
        # forward
        illumination, reflectance, noise = net(img_tensor)  # [1, c, h, w]
        # loss computing
        loss_recons = reconstruction_loss(img_tensor, illumination, reflectance, noise)
        loss_illu = illumination_smooth_loss(img_tensor, illumination)
        loss_reflect = reflectance_smooth_loss(img_tensor, illumination, reflectance)
        loss_noise = noise_loss(img_tensor, illumination, reflectance, noise)

        loss = loss_recons + config.illu_factor * loss_illu + config.reflect_factor * loss_reflect + config.noise_factor * loss_noise

        # backward
        net.zero_grad( )
        loss.backward(retain_graph=True)
        optimizer.step( )

        # log
        if i % 200 == 0:
            print("iter:", i, '  reconstruction loss:', float(loss_recons.data), '  illumination loss:',
                  float(loss_illu.data), '  reflectance loss:', float(loss_reflect.data), '  noise loss:',
                  float(loss_noise.data))

    # adjustment
    adjust_illu = torch.pow(illumination, config.gamma)
    res_image = adjust_illu * ((img_tensor - noise) / illumination)
    res_image = torch.clamp(res_image, min=0, max=1)

    if config.device != 'cpu':
        res_image = res_image.cpu()
        illumination = illumination.cpu()
        adjust_illu = adjust_illu.cpu()
        reflectance = reflectance.cpu()
        noise = noise.cpu()

    res_img = transforms.ToPILImage()(res_image.squeeze(0))
    #res_img.save('clean_image.jpg')
    return res_img


def dehaze_image(image_path,RRD=False,epoch_num=2):
    data_hazy = Image.open(image_path)

    #调整为4的倍数
    # 计算宽度和高度需要减去的像素数
    width_padding = data_hazy.width % 4
    height_padding = data_hazy.height % 4

    # 如果宽度或高度不是4的倍数，则添加额外的像素
    if width_padding != 0 or height_padding != 0:
        new_width = data_hazy.width - width_padding
        new_height = data_hazy.height - height_padding
        data_hazy = data_hazy.crop((0, 0, new_width, new_height))

    data_hazy = (np.asarray(data_hazy) / 255.0) #torch.Size([411, 550, 3])---->torch.Size([410, 550, 3])---->torch.Size([408, 548, 3])

    data_hazy = torch.from_numpy(data_hazy).float( ) #[1, 3, 408, 548]

    # import pdb
    # pdb.set_trace()

    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda( ).unsqueeze(0) #torch.Size([1, 3, 410, 550])

    # 下采样
    img_haze_2x = nn.MaxPool2d(2)(data_hazy)  #torch.Size([1, 3, 205, 275])
    img_haze_4x = nn.MaxPool2d(4)(data_hazy)  #torch.Size([1, 3, 102, 137])

    Epoch_num='Epoch'+str(epoch_num)+'.pth'



    # Scale3
    dehaze_net = net.dehaze_net(in_channels=3).cuda( )
    img_haze_4x = img_haze_4x.cuda( )
    #dehaze_net.load_state_dict(torch.load('snapshots/Scale3/'+Epoch_num))
    J_4x = dehaze_net(img_haze_4x, img_haze_4x)  #torch.Size([1, 3, 102, 137])
    upsample = nn.Upsample(scale_factor=2)
    J_4x_up = upsample(J_4x)  # 8  3 320 240

    # Scale2
    dehaze_net = net.dehaze_net(in_channels=6).cuda( )
    img_haze_2x = img_haze_2x.cuda( ) #torch.Size([1, 3, 205, 275])
    J_4x_up = J_4x_up.cuda( ) #torch.Size([1, 3, 204, 274])
    img_concat_2x = torch.cat((J_4x_up, img_haze_2x), dim=1)
    #dehaze_net.load_state_dict(torch.load('snapshots/Scale2/'+Epoch_num))
    J_2x = dehaze_net(img_concat_2x, img_haze_2x)
    J_2x_up = upsample(J_2x)

    # Scale1
    dehaze_net = net.dehaze_net(in_channels=6).cuda( )
    dehaze_net.load_state_dict(torch.load('snapshots/Scale1/'+Epoch_num))
    #dehaze_net.load_state_dict(torch.load('E:\wangchao\AODNet_2\snapshots\dehazer.pth'))
    img_concat_1x = torch.cat((data_hazy, J_2x_up), dim=1)
    clean_image = dehaze_net(img_concat_1x,data_hazy)#clean_image 是一个张量（tensor）对象

    # 拼接
    filename="results/" + image_path.split("/")[-1]
    #torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), filename)
    torchvision.utils.save_image(clean_image, filename)

    if RRD:
        #RRDNet
        Net = RRDNet( )
        Net = Net.to('cuda')
        res_img=pipline_retinex(Net, clean_image)
        res_img.save(filename)

    # 打开图像
    image = Image.open(filename)
    img1 = cv2.imread(image_path)
    # 计算前两个维度需要减去的元素数
    dim1_padding = img1.shape[0] % 4
    dim2_padding = img1.shape[1] % 4
    # 如果前两个维度不是4的倍数，则添加额外的元素
    if dim1_padding != 0 or dim2_padding != 0:
        new_dim1 = img1.shape[0] - dim1_padding
        new_dim2 = img1.shape[1] - dim2_padding
        img1 = img1[:new_dim1, :new_dim2, ...]

    img2 = cv2.imread(filename)

    # 计算结构相似度
    ssim_value = SSIM(img1, img2, channel_axis=2) #img1是雾图，img2是去雾后的图  #img1.shape (408, 548, 3)
    psnr_value = PSNR(img1, img2)

    # 添加文字
    # draw = ImageDraw.Draw(image)
    # text1 = "pnsr= {:.4f}".format(psnr_value )
    # text2 = "ssim = {:.4f}".format(ssim_value)
    # font = ImageFont.truetype("arial.ttf", 20)  # 设置字体和大小
    # textbbox1 = draw.textbbox((0, 0), text1, font=font)
    # textbbox2 = draw.textbbox((0, 0), text2, font=font)
    # x1 = (image.width - textbbox1[2]) / 2
    # y1 = image.height - textbbox1[3] - 10
    # x2 = (image.width - textbbox2[2]) / 2
    # y2 = y1 - textbbox2[3] - 10
    # draw.text((x1, y1), text1, font=font, fill=(255, 255, 255, 255))
    # draw.text((x2, y2), text2, font=font, fill=(255, 255, 255, 255))

    # 创建一个包含数据的列表
    data = [
        [image_path.split("\\")[-1], ssim_value, psnr_value],
    ]
    current_date = datetime.date.today( )
    formatted_date = current_date.strftime("%m_%d_")
    # 定义文件名
    shuju = f"{formatted_date}metrics.csv"
    # 打开一个CSV文件进行写操作
    with open('metricss/'+shuju, mode='a', newline='') as file:
        # 创建一个CSV写入器
        writer = csv.writer(file)
        # 将数据写入CSV文件
        for row in data:
            writer.writerow(row)

    # 保存图像
    image.save(filename)

    #拼接
    # # 加载两个图片
    # img1 = Image.open(filename)
    #
    # # 获取图片的宽度和高度
    # width1, height1 = img1.size
    # width2, height2 = res_img.size
    #
    # # 计算新图片的宽度和高度
    # new_width = width1 + width2
    # new_height = max(height1, height2)
    #
    # # 创建新的空白图片
    # new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    #
    # # 将原始图片拼接到新图片上
    # new_img.paste(img1, (0, 0))
    # new_img.paste(res_img, (width1, 0))

    # # 保存新图片
    # new_img.save("results/" + image_path.split("/")[-1])


if __name__ == '__main__':

    test_list = glob.glob("test_images/1/*")
    # 创建一个包含数据的列表
    current_date = datetime.date.today( )
    formatted_date = current_date.strftime("%m_%d_")
    shuju = f"{formatted_date}metrics.csv"
    header = ['Image', 'SSIM', 'PSNR']

    # 打开一个CSV文件进行写操作
    with open('metricss/'+shuju, mode='a', newline='') as file:
        # 创建一个CSV写入器
        writer = csv.writer(file)
        writer.writerow(header)

    folder_path = 'test_images/1'  # 文件夹路径
    image_count = 0  # 图片数量

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        try:
            with Image.open(os.path.join(folder_path, filename)) as img:
                image_count += 1  # 如果文件可以用Pillow库打开，就将图片数量加1
        except IOError:
            pass  # 如果文件不能用Pillow库打开，就跳过
    print('There are', image_count, 'images in the folder.')

    #一次
    number=0
    alltime=0.0
    Pic_num=0

    # 多次平均值
    allepoch = 50


    one=True
    if one:
        for image in test_list:
            start_time = time.time( )  # 记录开始时间
            number += 1
            Pic_num=number

            print('Pic[',number,'/',image_count,']==========>')
            #权重的次数
            Epoch_num =0
            dehaze_image(image,RRD=False,epoch_num=Epoch_num)

            end_time = time.time( )  # 记录结束时间
            elapsed_time = end_time - start_time  # 计算执行时间
            alltime= alltime+elapsed_time
            print('此图处理时间:', int(elapsed_time // 60), 'min',int(elapsed_time - (elapsed_time // 60) * 60), 's')
            print('目前总耗时：',  int(alltime // 60), 'min',int(alltime - (alltime // 60) * 60), 's')
            print(image, "done!")



        # # 读取csv文件
        # df = pd.read_csv(f"metricss/{formatted_date}metrics.csv")
        #
        # # 计算某一列的平均值
        # mean_value1 = df['SSIM'][-Pic_num:].astype(float).mean( )
        # print('Epoch',Epoch_num,'的SSIM平均值为：', mean_value1)
        # mean_value2 = df['PSNR'][-Pic_num:].astype(float).mean( )
        # print('Epoch',Epoch_num,'的PSNR平均值为：', mean_value2)
        # # 创建一个包含数据的列表
        # data = [
        #     ['均值', mean_value1, mean_value2],
        # ]
        # current_date = datetime.date.today( )
        # formatted_date = current_date.strftime("%m_%d_")
        # # 定义文件名
        # shuju = f"{formatted_date}metrics.csv"
        # # 打开一个CSV文件进行写操作
        # with open('metricss/' + shuju, mode='a', newline='') as file:
        #     # 创建一个CSV写入器
        #     writer = csv.writer(file)
        #     # 将数据写入CSV文件
        #     for row in data:
        #         writer.writerow(row)

    else:

        # 创建一个包含数据的列表
        current_date = datetime.date.today( )
        formatted_date = current_date.strftime("%m_%d_")
        shuju = f"{formatted_date}meanvalue.csv"
        header = ['num', 'SSIM', 'PSNR']
        # 打开一个CSV文件进行写操作
        with open('metricss/' + shuju, mode='a', newline='') as file:
            # 创建一个CSV写入器
            writer = csv.writer(file)
            writer.writerow(header)


        for i in range(allepoch):
            number = 0
            alltime = 0.0
            Pic_num = 0
            for image in test_list:
                start_time = time.time( )  # 记录开始时间
                number += 1
                Pic_num = number
                # 权重的次数
                Epoch_num = i
                dehaze_image(image, RRD=False, epoch_num=Epoch_num)

            import pandas as pd

            # 读取csv文件
            df = pd.read_csv(f"metricss/{formatted_date}metrics.csv")

            # 计算某一列的平均值
            ssim_mean_value = df['SSIM'][-Pic_num:].astype(float).mean( )
            print('Epoch', Epoch_num, '的SSIM平均值为：', ssim_mean_value)
            psnr_mean_value = df['PSNR'][-Pic_num:].astype(float).mean( )
            print('Epoch', Epoch_num, '的PSNR平均值为：', psnr_mean_value)

            # 创建一个包含数据的列表
            data = [
                ['Epoch'+ str(i), ssim_mean_value, psnr_mean_value],
            ]
            current_date = datetime.date.today( )
            formatted_date = current_date.strftime("%m_%d_")

            # 定义文件名
            shuju = f"{formatted_date}meanvalue.csv"
            # 打开一个CSV文件进行写操作
            with open('metricss/' + shuju, mode='a', newline='') as file:
                # 创建一个CSV写入器
                writer = csv.writer(file)
                # 将数据写入CSV文件
                for row in data:
                    writer.writerow(row)

