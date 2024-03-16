import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(orig_images_path, hazy_images_path):

	# # 图片文件夹路径
	# image_folder =  hazy_images_path
	#
	# # 遍历图片文件夹中的所有文件
	# for file_name in os.listdir(image_folder):
	# 	# 构造完整的文件路径
	# 	file_path = os.path.join(image_folder, file_name)
	# 	try:
	# 		# 打开图片文件
	# 		with Image.open(file_path) as img:
	# 			# 获取图片格式
	# 			image_format = img.format

	train_list = []
	val_list = []
	#单个文件夹
	image_list_haze = glob.glob(hazy_images_path + "*.png")
	#多个文件夹
	#image_list_haze = glob.glob(hazy_images_path + "**\\*.jpg")

	tmp_dict = {}

	for image in image_list_haze:
		image = image.split("/")[-1]  #'E:\\wangchao\\mist_dataset\\NHHAZE\\clean\\01.png'
		key = image.split("_")[0] + "_" + image.split("_")[1] + ".png" #'E:\\wangchao\\mist_dataset\\NHHAZE\\clean\\01.png.png'

		if key in tmp_dict.keys():
			tmp_dict[key].append(image)
		else:
			tmp_dict[key] = []
			tmp_dict[key].append(image)


	train_keys = []
	val_keys = []

	len_keys = len(tmp_dict.keys())
	for i in range(len_keys):
		if i < len_keys*9/10:
			train_keys.append(list(tmp_dict.keys())[i])
		else:
			val_keys.append(list(tmp_dict.keys())[i])


	for key in list(tmp_dict.keys()):
		if key in train_keys:
			for hazy_image in tmp_dict[key]:

				key = key.split("\\")[-1] #key='01.png.png'
				hazy_image =hazy_image.split("\\")[-1] #hazy_image='01.png'
				train_list.append([orig_images_path + key, hazy_images_path + hazy_image]) #[['E:\\wangchao\\mist_dataset\\NHHAZE\\clean\\01.png.png', 'E:\\wangchao\\mist_dataset\\NHHAZE\\01.png']]

		else:
			for hazy_image in tmp_dict[key]:
				key = key.split("\\")[-1]
				hazy_image = hazy_image.split("\\")[-1]
				val_list.append([orig_images_path + key, hazy_images_path + hazy_image])

	random.shuffle(train_list)
	random.shuffle(val_list)

	return train_list, val_list

	

class dehazing_loader(data.Dataset):

	def __init__(self, orig_images_path, hazy_images_path, mode='train'):

		self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)
		# self.train_list=hazy_images_path
		# self.val_list=orig_images_path

		# import pdb
		# pdb.set_trace()

		if mode == 'train':
			self.data_list = self.train_list
			print("Total training examples:", len(self.train_list))
		else:
			self.data_list = self.val_list
			print("Total validation examples:", len(self.val_list))



	def __getitem__(self, index):

		data_orig_path, data_hazy_path = self.data_list[index]

		data_orig = Image.open(data_orig_path)
		data_hazy = Image.open(data_hazy_path)

		data_orig = data_orig.resize((480,640), Image.ANTIALIAS)
		data_hazy = data_hazy.resize((480,640), Image.ANTIALIAS)

		data_orig = (np.asarray(data_orig)/255.0) 
		data_hazy = (np.asarray(data_hazy)/255.0) 

		data_orig = torch.from_numpy(data_orig).float()
		data_hazy = torch.from_numpy(data_hazy).float()

		return data_orig.permute(2,0,1), data_hazy.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)


if __name__ == '__main__':
	orig_images_path='E:\\wangchao\\mist_dataset\\RESIDE\\OTSBETA\\clear\\ '
	hazy_images_path='E:\\wangchao\\mist_dataset\\RESIDE\\OTSBETA\\allhazy\\part1\\ '
	dehazing_loader(orig_images_path, hazy_images_path)