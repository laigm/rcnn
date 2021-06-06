import math

import numpy as np
import tensorflow as tf
from PIL import Image


class BBoxUtility(object):
    def __init__(self, overlap_threshold=0.7, ignore_threshold=0.3, rpn_pre_boxes=6000, rpn_nms=0.7, classifier_nms=0.3,
                 top_k=300):
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self.rpn_pre_boxes = rpn_pre_boxes

        self.rpn_nms = rpn_nms
        self.classifier_nms = classifier_nms
        self.top_k = top_k

    def iou(self, box):
        """
            计算每个真实框与所有的先验框的 iou
        """
        # 判断真实框与先验框的重合情况
        # 真实框与先验框重合部分的左上角与右上角坐标信息
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        # 真实框与先验框重合部分的宽和高
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        # 真实框与先验框重合部分的面积，即（真实框 交 先验框）的面积
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # （真实框 U 先验框）的面积
        union = area_true + area_gt - inter
        # 计算iou
        iou = inter / union
        return iou

    def encode_ignore_box(self, box, return_iou=True):
        # box.shape = (4,)
        # iou.shape = (12996,)
        iou = self.iou(box)
        # ignored_box.shape = (12996, 1)
        ignored_box = np.zeros((self.num_priors, 1))
        # ---------------------------------------------------#
        #   找到处于忽略门限值范围内的先验框
        # ---------------------------------------------------#
        # assign_mask_ignore.shape = (12996,)   dtype = bool
        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        """
            slice_ignored_box.shape = (12996,)
            slice_ignored_box = ignored_box[:, 0]
        """
        # 将 iou 在 (0.3, 0.7) 范围内的先验框的交并比保存到 ignored_box 中
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]
        # encoded_box.shape = (12996,5)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        # ---------------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框     assign_mask.shape = (12996,)
        # ---------------------------------------------------#
        assign_mask = iou > self.overlap_threshold
        """
            numpy.any(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>):
                Test whether any array element along a given axis evaluates to True.
                The default (axis=None) is to perform a logical OR over all the dimensions of the input array.
        """
        # 如果没有使得 iou 大于 0.7 的先验框，则保存使得 iou 最大的先验框
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        # 将保留的重合度较高的先验框的交并比保存到 encoded_box 中
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        # 将上方所过滤出 iou 较大的先验框保存到 assigned_priors 中。  示例：assigned_priors.shape = (87, 4)
        assigned_priors = self.priors[assign_mask]
        # ---------------------------------------------#
        #   逆向编码，将真实框转化为FRCNN预测结果的格式
        #   先计算真实框的中心与长宽
        # ---------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # ---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        # ---------------------------------------------#
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] - assigned_priors[:, :2])
        # ------------------------------------------------#
        #   逆向求取efficientdet应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        """
        numpy.ravel(a, order='C')
            Return a contiguous flattened array.
        """
        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes, anchors):
        """
        boxes:真实框信息     boxes.shape = (图片中目标数目, 5)
        anchors:先验框信息   anchors.shape = (12996, 4)
        """
        # 计算有多少先验框
        self.num_priors = len(anchors)
        self.priors = anchors
        # ---------------------------------------------------#
        #   assignment分为2个部分
        #   assignment[:,:4] 的内容为网络应该有的回归预测结果
        #   assignment[:,4] 的内容为先验框是否包含物体，默认为背景
        # ---------------------------------------------------#
        assignment = np.zeros((self.num_priors, 4 + 1))

        # 预先设定 0 代表先验框内为背景
        assignment[:, 4] = 0.0
        # 如果真实框信息长度为0，即图片中没有目标，则直接返回 assignment
        if len(boxes) == 0:
            return assignment

        # ---------------------------------------------------#
        #   对每一个真实框都进行iou计算
        # ---------------------------------------------------#
        """
            def f(a):
                return (a[0]+a[1])*2
            b=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
            np.apply_along_axis(f,0,b) 
            # 结果:array([12, 16, 20, 24])
            # (1+5)*2=12  (2+6)*2=16依次类推
            np.apply_along_axis(f,1,b)
            # 结果:array([ 6, 22, 38])
            # (1+2)*2=6  (5+6)*2=22依次类推
        """
        """
        i 为图片中真实框的个数
            apply_along_axis_boxes.shape = (i, 2)
            apply_along_axis_boxes[0, 0].shape = (i, 64980)
            apply_along_axis_boxes[0, 1].shape = (i, 12996)
        """
        apply_along_axis_boxes = np.apply_along_axis(self.encode_ignore_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ingored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])

        # ---------------------------------------------------#
        #   在reshape后，获得的ingnored_boxes的shape为：
        #   [num_true_box, num_priors, 1] 其中1为iou
        # ---------------------------------------------------#
        """
            备忘录：以横轴代表12996个先验框，纵轴代表 num_true_box 个先验框来理解以下内容
        """
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        # ignored_iou.shape = (12996,)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        # ignored_iou_mask.shape = (12996,) dtype = bool
        ignore_iou_mask = ignore_iou > 0
        # 将置信度设为 -1 表示需要忽略的先验框
        assignment[:, 4][ignore_iou_mask] = -1
        # ---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_priors, 4+1]
        #   4是编码后的结果，1为iou
        # ---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        # ---------------------------------------------------#
        #   求取每一个先验框重合度最大的真实框
        # ---------------------------------------------------#
        # best_iou.shape = best_iou_idx.shape = best_iou_mask.shape = (num_priors,) = (12996,)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        # 求取iou较大的先验框集合中每一个先验框重合度最大的真实框
        # best_iou_idx.shape = (iou较大的先验框个数 , )
        best_iou_idx = best_iou_idx[best_iou_mask]

        # ---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        # ---------------------------------------------------#
        assign_num = len(best_iou_idx)
        # 将编码后的真实框取出    encoded_boxes.shape = (真实框个数, iou较大的先验框个数, 5)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # ----------------------------------------------------------#
        #  1 代表当前先验框包含目标
        # ----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox):
        """
        获得真实框的左上角和右下角
        """
        # 获得先验框的宽与高     (12996,)
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
        decode_bbox_center_y += prior_center_y
        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
        decode_bbox_height *= prior_height
        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        # 真实框的左上角与右下角进行堆叠   decode_bbox.shape = (12996, 4)
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out_rpn(self, predictions, mbox_priorbox):
        # ---------------------------------------------------#
        #   获得种类的置信度    (1, 12996, 1)
        # ---------------------------------------------------#
        mbox_conf = predictions[0]
        # ---------------------------------------------------#
        #   mbox_loc是回归预测结果     (1, 12996, 4)
        # ---------------------------------------------------#
        mbox_loc = predictions[1]
        # ---------------------------------------------------#
        #   获得网络的先验框，即 anchors  (12996, 4)
        # ---------------------------------------------------#
        mbox_priorbox = mbox_priorbox

        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(mbox_loc)):
            # --------------------------------#
            #   利用回归结果对先验框进行解码  decode_bbox.shape = (12996, 4)
            # --------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)
            # --------------------------------#
            #   取出先验框内包含物体的概率
            # --------------------------------#
            # c_confs.shape = (12996,)
            c_confs = mbox_conf[i, :, 0]
            """
            y = argsort(x),将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
                示例：
                    >> x = np.array([3, 1, 2])
                    >> np.argsort(x)
                    array([1, 2, 0])
            """
            # 从小到大排序，逆序保存索引（即保存从大到小排序的索引）
            argsort_index = np.argsort(c_confs)[::-1]
            # 保留概率最高的 6000 个框   c_confs.shape = (6000,)     decode_bbox.shape = (6000, 4)
            c_confs = c_confs[argsort_index[:self.rpn_pre_boxes]]
            decode_bbox = decode_bbox[argsort_index[:self.rpn_pre_boxes], :]
            """
            tf.image.non_max_suppression(
                boxes, scores, max_output_size, iou_threshold=0.5,
                score_threshold=float('-inf'), name=None)
            """
            # 进行iou的非极大抑制   idx.shape = (300,)
            idx = tf.image.non_max_suppression(decode_bbox, c_confs, self.top_k, iou_threshold=self.rpn_nms).numpy()
            # 取出在非极大抑制中效果较好的内容      good_boxes.shape = (300, 4)     confs.shape = (300,1)
            good_boxes = decode_bbox[idx]
            confs = c_confs[idx][:, None]
            # 水平方向堆叠 confs 和 good_boxes
            c_pred = np.concatenate((confs, good_boxes), axis=1)
            argsort = np.argsort(c_pred[:, 0])[::-1]
            # 按概率从大到小排序保存这些建议框
            c_pred = c_pred[argsort]
            results.append(c_pred)

        return np.array(results)

    def detection_out_classifier(self, predictions, proposal_box, config, confidence):
        # ---------------------------------------------------#
        #   获得种类的置信度    proposal_conf.shape = (1, 300, 21)
        # ---------------------------------------------------#
        proposal_conf = predictions[0]
        # ---------------------------------------------------#
        #   proposal_loc是回归预测结果     proposal_loc.shape = (1, 300, 80)
        # ---------------------------------------------------#
        proposal_loc = predictions[1]

        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(proposal_conf)):
            proposal_pred = []
            # 将 proposal_box[i, :, 2]、proposal_box[i, :, 3]的信息更改为建议框的宽和高
            proposal_box[i, :, 2] = proposal_box[i, :, 2] - proposal_box[i, :, 0]
            proposal_box[i, :, 3] = proposal_box[i, :, 3] - proposal_box[i, :, 1]
            # 遍历每张图片所对应的的 300 个建议框
            for j in range(proposal_conf[i].shape[0]):
                # 舍弃置信度小于 confidence 的框
                if np.max(proposal_conf[i][j, :-1]) < confidence:
                    continue
                # 获得单个框回归预测结果所属分类的索引
                label = np.argmax(proposal_conf[i][j, :-1])
                # 获得单个框回归预测结果所属分类的置信度
                score = np.max(proposal_conf[i][j, :-1])
                # 建议框的坐标信息
                (x, y, w, h) = proposal_box[i, j, :]
                # 建议框的调整参数
                (tx, ty, tw, th) = proposal_loc[i][j, 4 * label: 4 * (label + 1)]
                tx /= config.classifier_regr_std[0]
                ty /= config.classifier_regr_std[1]
                tw /= config.classifier_regr_std[2]
                th /= config.classifier_regr_std[3]
                # 原始建议框的中心
                cx = x + w / 2.
                cy = y + h / 2.
                # 调整后建议框的中心
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                # 调整后的建议框的宽和高
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h
                # 调整后建议框的左上角
                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.
                # 调整后建议框的右下角
                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                proposal_pred.append([x1, y1, x2, y2, score, label])

            num_classes = np.shape(proposal_conf)[-1]  # 21
            # proposal_pred.shape = (84, 6)     其中参数 84 不定
            # 对框进行筛选，并对建议框进行调整后的建议框信息
            proposal_pred = np.array(proposal_pred)
            good_boxes = []
            if len(proposal_pred) != 0:
                for c in range(num_classes):
                    # mask.shape = (84,)    dtype = bool
                    mask = proposal_pred[:, -1] == c
                    if len(proposal_pred[mask]) > 0:
                        boxes_to_process = proposal_pred[:, :4][mask]
                        confs_to_process = proposal_pred[:, 4][mask]
                        """
                        def non_max_suppression(boxes, scores, max_output_size, iou_threshold=0.5, 
                                                        score_threshold=float('-inf'), name=None):
                        """
                        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, self.top_k,
                                                           iou_threshold=self.classifier_nms).numpy()
                        # 取出在非极大抑制中效果较好的内容
                        # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                        good_boxes.extend(proposal_pred[mask][idx])
            results.append(good_boxes)

        return results
