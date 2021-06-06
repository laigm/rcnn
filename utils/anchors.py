import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.config import Config

config = Config()


def generate_anchors(sizes=None, ratios=None):
    if sizes is None:
        # sizes = [128, 256, 512]
        sizes = config.anchor_box_scales
    if ratios is None:
        # rations = [[1, 1], [1, 2], [2, 1]]
        ratios = config.anchor_box_ratios
    num_anchors = len(sizes) * len(ratios)  # 3*3=9
    anchors = np.zeros((num_anchors, 4))
    """
    anctros = [[  0.   0. 128. 128.],
               [  0.   0. 256. 256.],
               [  0.   0. 512. 512.],
               [  0.   0. 128. 128.],
               [  0.   0. 256. 256.],
               [  0.   0. 512. 512.],
               [  0.   0. 128. 128.],
               [  0.   0. 256. 256.],
               [  0.   0. 512. 512.]]
    """
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    """
    anctros = [[   0.    0.  128.  128.], 
               [   0.    0.  256.  256.], 
               [   0.    0.  512.  512.], 
               [   0.    0.  128.  256.], 
               [   0.    0.  256.  512.], 
               [   0.    0.  512. 1024.], 
               [   0.    0.  256.  128.], 
               [   0.    0.  512.  256.], 
               [   0.    0. 1024.  512.]]
    """
    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]
    """
    anctros = [[ -64.    0.   64.  128.], 
               [-128.    0.  128.  256.], 
               [-256.    0.  256.  512.], 
               [ -64.    0.   64.  256.], 
               [-128.    0.  128.  512.], 
               [-256.    0.  256. 1024.], 
               [-128.    0.  128.  128.], 
               [-256.    0.  256.  256.], 
               [-512.    0.  512.  512.]]
    """
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    """
    anctros = [[ -64.  -64.   64.   64.], 
               [-128. -128.  128.  128.], 
               [-256. -256.  256.  256.], 
               [ -64. -128.   64.  128.], 
               [-128. -256.  128.  256.], 
               [-256. -512.  256.  512.], 
               [-128.  -64.  128.   64.], 
               [-256. -128.  256.  128.], 
               [-512. -256.  512.  256.]]
    """
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, anchors, stride=config.rpn_stride):
    # shift_x = shift_y = [  8.  24.  40.  56.  72.  88. 104. 120. 136. 152. 168. 184.
    #                       200. 216., 232. 248. 264. 280. 296. 312. 328. 344. 360. 376.
    #                       392. 408. 424. 440., 456. 472. 488. 504. 520. 536. 552. 568. 584. 600.]
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    # (38,38)
    # shift_x: 纵向复制
    # shift_y: 转置后水平复制
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 平铺 (1444,)
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    # (4, 1444)
    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    # (1444, 4)
    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]    # 9
    k = np.shape(shifts)[0]     # 1444
    # (1444, 9, 4)
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    # (12996, 4)
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors


def get_anchors(shape, width, height):
    """
    获得 38*38*9 个anchors
    """
    """
    shape: 共享特征层尺寸，示例为 [38, 38]
    width: resize后图片的宽,示例为 600
    height: resize后图片的高，示例为 600
    anchors = [[ -64.  -64.   64.   64.],
               [-128. -128.  128.  128.],
               [-256. -256.  256.  256.],
               [ -64. -128.   64.  128.],
               [-128. -256.  128.  256.],
               [-256. -512.  256.  512.],
               [-128.  -64.  128.   64.],
               [-256. -128.  256.  128.],
               [-512. -256.  512.  256.]]
    """
    anchors = generate_anchors()
    network_anchors = shift(shape, anchors)
    # 以下为归一化处理，将值范围固定到 [0, 1]
    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height
    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors
