import cv2
from format import format_label
import sys
import numpy as np
from model import Yolo
from config import *
from nms import non_max_suppression

# C = 9
# S = 7
# B = 2
# target_size = 448
# origin_w = 1242
# origin_h = 375
scale_x = float(origin_size[0]) / target_size[0]
scale_y = float(origin_size[1]) / target_size[1]
# CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)


def display(img_in, img_out, y_pred, obj_threshold=0.3):
    y_pred[..., 4]  = _sigmoid(y_pred[..., 4])
    y_pred[..., 5:] = y_pred[..., 4][..., np.newaxis] * _softmax(y_pred[..., 5:])
    y_pred[..., 5:] *= y_pred[..., 5:] > obj_threshold
    img = cv2.imread(img_in)
    cell_size = target_size[0] / S
    boxes = []
    for row, cell_row in enumerate(y_pred):
        for col, cell in enumerate(cell_row):
            # find bbox with object
            for b, obj_abox in enumerate(cell):
                if obj_abox[4] < obj_threshold:
                    continue
                # TODO:
                # x = (col + obj_abox[C + 1]) * cell_size
                # y = (row + obj_abox[C + 2]) * cell_size
                # w = obj_abox[C + 3] * target_size[0]
                # h = obj_abox[C + 4] * target_size[0]
                x = (col + _sigmoid(obj_abox[0])) * cell_size
                y = (row + _sigmoid(obj_abox[1])) * cell_size
                w = anchor_boxes[b][0] * np.exp(obj_abox[2]) * cell_size # unit: image width
                h = anchor_boxes[b][1] * np.exp(obj_abox[3]) * cell_size # unit: image height
                #print 'x: {0} y: {1} w: {2} h: {3} row: {4} col: {5}'.format(x, y, w, h, row, col)
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                # rescale
                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                #print '({0}, {1}) ({2}, {3})'.format(x1, y1, x2, y2)
                boxes.append([x1, y1, x2, y2, np.max(obj_abox[5:]), int(np.argmax(obj_abox[5:]))])
    boxes = non_max_suppression(np.array(boxes), 0.3)
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # class
        cls = box[5]
        cv2.putText(img, CLASSES[cls], (box[0], box[1]), font, 0.5, (0,0,255), 1)
    cv2.imwrite(img_out, img)

if __name__ == '__main__':
    '''
    labels = format_label('../KITTI/training/split_0.1/train/label')
    key = sys.argv[1]
    img_path_in = '../KITTI/training/split_0.1/train/image/0/%s.png' % key
    img_path_out = './%s.png' % key
    display(img_path_in, img_path_out, labels[key])
    '''

    img_in = sys.argv[1]
    img_out = sys.argv[2]
    yolo = Yolo()
    yolo.load_pretrained_weights('../keras-yolo2/full_yolo_backend.h5')
    yolo.load_weights('../keras-yolo2/full_yolo_kitti.h5')
    # yolo.load_pretrained_weights('./yolo.weights')
    y = yolo.predict(img_in)
    display(img_in, img_out, y[0])
