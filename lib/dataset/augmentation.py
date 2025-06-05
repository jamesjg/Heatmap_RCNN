"""
    data augmentation: Rotate, Flip, occlusion, blur, translate
                    optional: scale noise brightness sharpen contrast
"""

import os
from PIL import Image,ImageFilter
import numpy as np
import random
from math import floor,ceil

# !! Must do deep copy before data augmentation

"""
    args:
        image: type is PIL.image 
        target: numpy array type, (landmark_num,2)
"""

def random_translate(image, target):
    # PIL type
    if random.random() > 0.5:
        image_height, image_width = image.size
        a = 1
        b = 0
        c = int((random.random()-0.5) * 60)
        d = 0
        e = 1
        f = int((random.random()-0.5) * 60)
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        target[:, 0] -= 1.*c/image_width
        target[:, 1] -= 1.*f/image_height
        return image, target
    else:
        return image, target


def random_blur(image):
    # PIL type
    if random.random() > 0.7 :
        image = image.filter(ImageFilter.GaussianBlur(random.random()*5))
    return image

def random_flip(image,target,points_flip):
    # PIL type
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target = target[points_flip, :]
        target[:,0] = 1-target[:,0]
        return image, target
    else:
        return image, target

def random_rotate(image, target, angle_max):
    # PIL type
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_num= target.shape[0]
        target_center = target - np.array([center_x, center_y])
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c,-s), (s, c)))
        target_center_rot = np.matmul(target_center, rot)
        target_rot = target_center_rot + np.array([center_x, center_y])
        return image, target_rot
    else:
        return image, target


def random_occlusion(image):
    # Need change data type
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height*0.4*random.random())
        occ_width = int(image_width*0.4*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np.astype('uint8'), 'RGB')
        return image_pil
    else:
        return image

def pad_crop(image,target):
    """
        add pad for the overflow points
        image need change data type
        border_pad : 8px
    """
    image_height, image_width = image.size

    l,t = np.min(target,axis=0)
    r,b = np.max(target,axis=0)
    
    # if the over border is left than grid_size, pass
    grid_size = 0.5 / image_height 
    
    if l > -grid_size and t > -grid_size and r < (1 + grid_size) and b < (1 + grid_size):
        target = np.maximum(target,0)
        target = np.minimum(target,1)
        return image,target
    border_pad_value = 8
    image_np = np.array(image).astype(np.uint8)
    border_size = np.zeros(4).astype('int') # upper bottom left right
    if l < 0:
        border_size[2] = ceil(-l * image_height) + border_pad_value #left
    if t < 0:
        border_size[0] = ceil(-t * image_width) + border_pad_value #upper
    if r > 1:
        border_size[3] = ceil((r-1) * image_height) + border_pad_value #right
    if b > 1:
        border_size[1] = ceil((b-1) * image_width) + border_pad_value #bottom
    border_img = np.zeros((image_width  + border_size[0] + border_size[1],
                           image_height + border_size[2] + border_size[3], 3)).astype(np.uint8)

    border_img[border_size[0] : border_size[0]+image_height, 
               border_size[2] : border_size[2]+image_width,:] = image_np
               
    image_pil = Image.fromarray(border_img.astype('uint8'), 'RGB')
    image_pil = image_pil.resize((image_height,image_width))
    target = (target * np.array([image_height,image_width]) + 
              np.array([border_size[2],border_size[0]])) /  np.array([border_img.shape[1],border_img.shape[0]])

    return image_pil, target

def ignore_crop(target):
    target = np.maximum(target,0)
    target = np.minimum(target,1)
    return target


def flip_points(data_type="WFLW"):
    data_type = data_type.upper()
    points_flip = None
    if data_type == '300W':
        points_flip = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65]
        assert len(points_flip) == 68
    elif data_type == 'WFLW':
        points_flip = [32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,46,45,44,43,42,50,49,48,47,37,36,35,34,33,41,40,39,38,51,52,53,54,59,58,57,56,55,72,71,70,69,68,75,74,73,64,63,62,61,60,67,66,65,82,81,80,79,78,77,76,87,86,85,84,83,92,91,90,89,88,95,94,93,97,96]
        assert len(points_flip) == 98
    elif data_type == 'COFW':
        points_flip = [1,0,3,2,6,7,4,5,9,8,11,10,14,15,12,13,17,16,19,18,20,21,23,22,24,25,26,27,28]
        assert len(points_flip) == 29
    elif data_type == 'AFLW':
        points_flip = [5,4,3,2,1,0,11,10,9,8,7,6,14,13,12,17,16,15,18]
        assert len(points_flip) == 19
    else:
        print('No such data!')
        exit(0)
    return points_flip


def check_size(imgs,config):
    """
        check dataset read success and size is same as input size.
        Args:
            imgs: [Image,...]
            config : config py
    """
    for i in range(len(imgs)):
        img = imgs[i] 
        if img.height != config.data.input_size or img.width != config.data.input_size:
            print("{}th Image is not applicable ({},{}),need delete or resize.".format(i+1,img.height,img.width))
            exit(-1)

