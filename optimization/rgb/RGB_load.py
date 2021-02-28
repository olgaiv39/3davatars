import tensorflow as tf
import numpy as np
import cv2
import sys
import os

sys.path.append("../..")

from utils.basis import load_3dmm_basis, get_geometry, get_region_uv_texture

from PIL import Image


class RGB_load(object):
    @staticmethod
    def load_landmark(path, num_lms):
        print("load lm:", path)
        f = open(path)
        arr = f.readlines()
        landmarks = []
        images = []
        for one in arr:
            # print(one)
            splits = one.strip().split(" ")
            imgname = splits[0]  # .split('/')[-1]
            # imgnumber = splits[0].split('/')[-1].split('.')[0]

            lm = [float(n) for n in splits[1 : 1 + num_lms * 2]]
            lm = np.array(lm)
            if lm.shape[0] < num_lms:
                continue
            lm = np.reshape(lm, (num_lms, 2))
            landmarks.append(lm)
            images.append(imgname)
        return landmarks, images

    @staticmethod
    def load_rgb_data(base_dir, project_type, num_of_img):

        # load data
        lmk3d_path = os.path.join(base_dir, "lmk_3D_86pts.txt")
        lmk2d_path = os.path.join(base_dir, "lmk_2D_68pts.txt")
        lmk3d_list, images_name_list = RGB_load.load_landmark(lmk3d_path, 86)
        lmk2d_list, _ = RGB_load.load_landmark(lmk2d_path, 68)

        img_triplet = []
        img_ori_triplet = []
        seg_list_triplet = []
        lmk3d_triplet = []
        lmk2d_triplet = []

        for index in range(0, num_of_img):
            img_name = images_name_list[index]
            img_path = os.path.join(base_dir, img_name)
            img_ori_path = os.path.join(
                base_dir, img_name[:-4] + "_ori" + img_name[-4:]
            )
            seg_path = os.path.join(base_dir, img_name[0:-4] + ".npy")

            img = np.asarray(Image.open(img_path)).astype(np.float32)
            img_ori = np.asarray(Image.open(img_ori_path)).astype(np.float32)
            lmk3d = lmk3d_list[index]
            lmk2d = lmk2d_list[index]
            seg = np.reshape(np.load(seg_path), (img.shape[0], img.shape[1]))
            seg_list = []
            for i in range(19):
                seg_list.append((seg == i).astype(np.float32))

            seg_list = np.stack(seg_list, axis=2)

            img_triplet.append(img)
            img_ori_triplet.append(img_ori)
            lmk3d_triplet.append(lmk3d)
            lmk2d_triplet.append(lmk2d)
            seg_list_triplet.append(seg_list)

        info = {
            "img_list": np.array(img_triplet),
            "img_ori_list": np.array(img_ori_triplet),
            "lmk_list3D": np.array(lmk3d_triplet),
            "lmk_list2D": np.array(lmk2d_triplet),
            "seg_list": np.array(seg_list_triplet),
        }

        if project_type == "Pers":
            se3_init = [[0.0], [np.pi], [0.0], [0.0], [0.0], [50.0]]
            info["K"] = np.array([[[-500.0, 0, 150.0], [0, -500.0, 150.0], [0, 0, 1]]])

        elif project_type == "Orth":
            se3_init = [[0.0], [np.pi], [0.0], [0.0], [0.0], [10.0]]
            info["K"] = np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
        else:
            raise Exception("project_type not in [Pers, Orth]!")

        info["se3_list"] = np.array([se3_init] * num_of_img)

        return info
