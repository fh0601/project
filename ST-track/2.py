# from https://blog.csdn.net/sinat_33486980/article/details/105684839?spm=1001.2014.3001.5502
# -*- coding:utf-8 -*-
import os
import cv2

'''
显示跟踪训练数据集标注
'''
# root_path="E:\XYL\dataset\\visdrone2019-MOT\VisDrone2019-MOT-test-dev"  # VisDrone数据集
root_path = r"E:\XYL\dataset\\UAVDT_M"  # UAVDT 数据集
img_dir = "images\\test"
label_dir = "labels_with_ids\\test"

imgs = os.listdir(root_path + "/" + img_dir)
for i, img in enumerate(imgs):
    # img_name=img[:-4]  # for MOT
    img_name = img
    if 'visdrone' in root_path:
        label_path = os.path.join(root_path + "/" + label_dir + "/" + img_name + "/" + "0000001.txt")  # 可视化第一帧的标签
    elif 'UAVDT' in root_path:  # 区分大小写
        label_path = os.path.join(root_path + "/" + label_dir + "/" + img_name + "/img1/" + "img000001.txt")
    label_f = open(label_path, "r")
    lines = label_f.readlines()
    if 'visdrone' in root_path:
        img_path = os.path.join(root_path + "/" + img_dir + "/" + img_name + "/" + "0000001.jpg")  # 没有img1文件夹，命名是7位数
    elif 'UAVDT' in root_path:  # 区分大小写
        img_path = os.path.join(
            root_path + "/" + img_dir + "/" + img_name + "/img1/" + "img000001.jpg")  # 有img1文件夹，命名是6位数
    img_data = cv2.imread(img_path)
    H, W, C = img_data.shape
    for line in lines:
        line_list = line.strip().split()
        class_num = int(line_list[0])  # 类别号
        obj_ID = int(line_list[1])  # 目标ID
        x, y, w, h = line_list[2:]  # 中心坐标，宽高（经过原图宽高归一化后）
        x = int(float(x) * W)
        y = int(float(y) * H)
        w = int(float(w) * W)
        h = int(float(h) * H)
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = left + w
        bottom = top + h
        cv2.circle(img_data, (x, y), 1, (0, 0, 255))
        cv2.rectangle(img_data, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img_data, str(obj_ID), (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    resized_img = cv2.resize(img_data, (W, H))
    cv2.imshow("label", resized_img)
    cv2.waitKey(1000)