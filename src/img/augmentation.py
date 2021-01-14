import cv2
import numpy as np


def random_shift_scale_rotate(image, angle, scale, aspect, shift_dx, shift_dy,
                              borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if len(image.shape) == 3:  # Img or mask
            height, width, channels = image.shape
        else:
            height, width = image.shape

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(shift_dx * width)
        dy = round(shift_dy * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx  # 计算cos对应x的变换，乘x放缩尺度   # [cos, -sin]
        ss = np.math.sin(angle / 180 * np.math.pi) * sy  #                               # [sin, cos]
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])  # 旋转矩阵

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2]) # 将[0, 0]变为中心，四个顶点位置
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy]) # 四个顶点位置旋转放缩后，去中心化平移回去，并shift漂移几个位置。

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1) # 得到方框0到方框1的变换矩阵？

        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0, 0)) # 边界以0填充？
    return image


def random_horizontal_flip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)  # 1表示水平翻转、0表示垂直翻转
        mask = cv2.flip(mask, 1)

    return image, mask


def augment_img(img, mask):  # 对img和mask同时进行几个操作
    rotate_limit = (-45, 45)
    aspect_limit = (0, 0)
    scale_limit = (-0.1, 0.1) # 放缩倍数
    shift_limit = (-0.0625, 0.0625)
    shift_dx = np.random.uniform(shift_limit[0], shift_limit[1])
    shift_dy = np.random.uniform(shift_limit[0], shift_limit[1])
    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
    scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1]) # [0.9, 1.1]倍之间
    aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1]) # 1

    img = random_shift_scale_rotate(img, angle, scale, aspect, shift_dx, shift_dy)
    mask = random_shift_scale_rotate(mask, angle, scale, aspect, shift_dx, shift_dy)

    img, mask = random_horizontal_flip(img, mask)
    return img, mask
