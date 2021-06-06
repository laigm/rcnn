import random
import time
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.anchors import get_anchors


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def cls_loss(ratio=3):
    def _cls_loss(y_true, y_pred):
        # ---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 1]
        #   y_pred [batch_size, num_anchor, 1]
        # ---------------------------------------------------#
        labels = y_true
        # ---------------------------------------------------#
        #   -1 是需要忽略的, 0 是背景, 1 是存在目标
        # ---------------------------------------------------#
        anchor_state = y_true
        classification = y_pred

        # ---------------------------------------------------#
        #   获得无需忽略的所有样本
        # ---------------------------------------------------#
        indices_for_no_ignore = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels_for_no_ignore = tf.gather_nd(labels, indices_for_no_ignore)
        classification_for_no_ignore = tf.gather_nd(classification, indices_for_no_ignore)

        cls_loss_for_no_ignore = keras.backend.binary_crossentropy(labels_for_no_ignore, classification_for_no_ignore)
        cls_loss_for_no_ignore = keras.backend.sum(cls_loss_for_no_ignore)

        # ---------------------------------------------------#
        #   进行标准化
        # ---------------------------------------------------#
        normalizer_no_ignore = tf.where(keras.backend.not_equal(anchor_state, -1))
        normalizer_no_ignore = keras.backend.cast(keras.backend.shape(normalizer_no_ignore)[0], keras.backend.floatx())
        normalizer_no_ignore = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_no_ignore)

        # 总的loss
        loss = cls_loss_for_no_ignore / normalizer_no_ignore
        return loss

    return _cls_loss


def smooth_l1(sigma=1.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # ---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 4+1]
        #   y_pred [batch_size, num_anchor, 4]
        # ---------------------------------------------------#
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # 找到正样本
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算smooth L1损失
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # 将所获得的loss除上正样本的数量
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        regression_loss = keras.backend.sum(regression_loss) / normalizer
        return regression_loss

    return _smooth_l1


def class_loss_regr(num_classes):
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4 * K.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])
        return loss

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    loss = K.mean(K.categorical_crossentropy(y_true, y_pred))
    return loss


def get_new_img_size(width, height, img_min_side=600):
    """
    将输入图片大小进行 resize，把最短边 resize 到 600px
    """
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2
        for i in range(4):
            # input_length = (input_length - filter_size + stride) // stride
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width), get_output_length(height)


class Generator(object):
    def __init__(self, bbox_util, train_lines, num_classes, Batch_size, input_shape=[600, 600], num_regions=256):
        self.bbox_util = bbox_util
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.num_classes = num_classes
        self.Batch_size = Batch_size
        self.input_shape = input_shape
        # 正、负样本数目之和
        self.num_regions = num_regions

    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """
            r实时数据增强的随机预处理
        """
        # 将annotation_line 按空格分割为一个数组
        # line[0] 为图片路径，line[0]后面的为：坐标信息+目标所属类别对应的索引
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        # 输入图片进行 resize 后的宽和高
        w, h = self.input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        return image_data, box_data

    def generate(self):
        while True:
            # 将 2007_train.txt 中属于训练集的部分按行打乱
            shuffle(self.train_lines)
            # 将打乱后的顺序保存到 lines 中
            lines = self.train_lines

            inputs = []
            target0 = []
            target1 = []
            target2 = []
            for annotation_line in lines:
                # img 数据增强后的图片      img.shape = (600, 600, 3)
                # y 图片里存放的信息（真实框）   示例：y.shape = (4, 5) 其中数字 4 代表目标个数，数字 5 代表：4个坐标信息+1个类别信息
                img, y = self.get_random_data(annotation_line)
                height, width, _ = np.shape(img)
                # 如果图片内有目标，则进行 if 语句块操作
                if len(y) > 0:
                    # 真实框的坐标信息
                    boxes = np.array(y[:, :4], dtype=np.float32)
                    # 对真实框进行归一化处理
                    boxes[:, 0] = boxes[:, 0] / width
                    boxes[:, 1] = boxes[:, 1] / height
                    boxes[:, 2] = boxes[:, 2] / width
                    boxes[:, 3] = boxes[:, 3] / height
                    y[:, :4] = boxes[:, :4]
                # 38*38*9个anchors   anchors.shape = (12996, 4)
                anchors = get_anchors(get_img_output_length(width, height), width, height)
                # ---------------------------------------------------#
                #   assignment分为2个部分，它的shape为 (12996, 5)
                #   :, :4      的内容为网络应该有的回归预测结果
                #   :,  4      的内容为先验框是否包含物体，默认为背景(1为包含物体，0为背景，-1为忽略的先验框)
                # ---------------------------------------------------#
                assignment = self.bbox_util.assign_boxes(y, anchors)

                classification = assignment[:, 4]
                regression = assignment[:, :]

                # ---------------------------------------------------#
                #   对正样本与负样本进行筛选，训练样本总和为256
                # ---------------------------------------------------#
                # 12996个先验框中，与真实框的iou较大的先验框（即被判断包含物体的先验框）个数
                mask_pos = classification[:] > 0
                # 正样本个数
                num_pos = len(classification[mask_pos])
                if num_pos > self.num_regions / 2:
                    val_locs = random.sample(range(num_pos), int(num_pos - self.num_regions / 2))
                    temp_classification = classification[mask_pos]
                    temp_regression = regression[mask_pos]
                    temp_classification[val_locs] = -1
                    temp_regression[val_locs, -1] = -1
                    classification[mask_pos] = temp_classification
                    regression[mask_pos] = temp_regression

                mask_neg = classification[:] == 0
                num_neg = len(classification[mask_neg])
                mask_pos = classification[:] > 0
                num_pos = len(classification[mask_pos])
                if len(classification[mask_neg]) + num_pos > self.num_regions:
                    val_locs = random.sample(range(num_neg), int(num_neg + num_pos - self.num_regions))
                    temp_classification = classification[mask_neg]
                    temp_classification[val_locs] = -1
                    classification[mask_neg] = temp_classification

                # 输入，增强后图片
                inputs.append(np.array(img))
                target0.append(np.reshape(classification, [-1, 1]))
                target1.append(np.reshape(regression, [-1, 5]))
                # 真实框
                target2.append(y)

                if len(inputs) == self.Batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = [np.array(target0, np.float32), np.array(target1, np.float32)]
                    tmp_y = target2
                    yield preprocess_input(tmp_inp), tmp_targets, tmp_y
                    inputs = []
                    target0 = []
                    target1 = []
                    target2 = []
