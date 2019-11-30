import os
import random
import numpy as np
import cv2 as cv


class Convert:
    """
    本类实现以下功能
    1、图片大小的修改
    2、路径的改变
    3、文件夹下图片的批量重命名
    """
    def __init__(self, path):
        """定义需要操作的文件夹"""
        self.__input_path = path
        self.__files = os.listdir(self.__input_path)
        if not self.__files:
            print("请检查输入目录是否正确")
            return
        self.__size = len(self.__files)

    def rename_image(self, output_path, start):
        """
        1、对输入文件夹下的所有图片进行重命名
        2、命名方式为数字，其中start为起始数字，之后依次+1 start为终止数字
        :param output_path: 输出文件夹位置
        :param start: 命名起始数字
        """
        for file_name in self.__files:
            print("当前正在修改图片名为{}".format(file_name))
            try:
                file_compose = file_name.split('.')
            except:
                continue
            old_path = os.path.join(self.__input_path, file_name)
            # new_path = os.path.join(output_path, '{}.{}'.format(start, file_compose[1]))#命名为数字
            new_path = os.path.join(output_path, file_name)#命名为原文件
            os.rename(old_path, new_path)
            start += 1
        print('done')

    def random_extract(self, output_path, start, number):
        """
        1、从输入文件夹随机抽取指定数量的图片并放入到输出文件夹
        2、将随机抽取的文件重新命名为以start为起始数字，依次+1
        :param output_path: 输出文件夹位置
        :param start: 命名起始数字
        :param number: 随机抽取的数量
        """
        if number > self.__size:
            print("抽取文件数量过多")
            return
        l = list(range(0, self.__size))
        random.shuffle(l)#定义随机数组
        for i in range(number):
            file_name = self.__files[l[i]]
            print("当前正在移动图片名为{}".format(file_name))
            try:
                file_compose = file_name.split('.')
            except:
                continue
            old_path = os.path.join(self.__input_path, file_name)
            # new_path = os.path.join(output_path, '{}.{}'.format(start, file_compose[1]))#命名为数字
            new_path = os.path.join(output_path, file_name)#命名为原文件
            os.rename(old_path, new_path)
            start += 1
        print("done")

    def convert_channels(self, output_path):
        """
        1、改变输入图片的通道
        2、输出命名为BGR+原名
        :param output_path:输出文件夹位置
        """
        for file_name in self.__files:
            print("当前正在转换图片名为{}".format(file_name))
            img_path = os.path.join(self.__input_path, file_name)
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            new_path = os.path.join(output_path, 'BGR_{}'.format(file_name))
            cv.imwrite(new_path, img)

    def rotate_img(self, output_path):
        """
        1、旋转图片
        2、图片命名为Rot+原名
        :param output_path: 输出文件夹位置
        :param rotation: 旋转角度
        """
        for file_name in self.__files:
            print("当前正在旋转图片名为{}".format(file_name))
            img_path = os.path.join(self.__input_path, file_name)
            img = cv.imread(img_path)
            # 顺时针旋转90度
            # img = np.rot90(img)
            # # 顺时针旋转180度
            # img = np.rot90(img, 2)
            # # 逆时针旋转90
            img = np.rot90(img, -1)
            new_path = os.path.join(output_path, 'Rot1_{}'.format(file_name))
            cv.imwrite(new_path, img)

    # def video2image(self, video_name, output_path):









# input_dir = 'D:/Zchao/KP_ResNet_Hat/Hat_dataset/old/train_test/0/'
# input_dir = 'D:/Zchao/KP_ResNet_Hat/val/0/'
input_dir = 'F:\Zchao\Gj_anno\dataset\json'
output_dir = 'F:\Zchao\Gj_anno\dataset\json_test'

convert = Convert(input_dir)
# convert.convert_channels(output_dir)
convert.random_extract(output_dir, 0, 40)
# convert.rotate_img(output_dir)
# convert.rename_image(output_dir, 0)