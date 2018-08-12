import cv2
from format import format_label
import sys
import numpy as np
from model import Yolo

C = 9
S = 7
B = 2
target_size = 448
origin_w = 1242
origin_h = 375
scale_x = float(origin_w) / target_size
scale_y = float(origin_h) / target_size
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)


def display(img_in, img_out, y_pred):
    y_pred = _sigmoid(y_pred)
    img = cv2.imread(img_in)
    cell_size = target_size / S
    print np.max(y_pred[..., C])
    for i, cell in enumerate(y_pred):
        # find bbox with object
        try:
            obj_abox = next(abox for abox in cell if abox[C] > 0.5)
        except:
            continue
        if obj_abox is None:
            continue
        row = i // S
        col = i % S
        #print obj_abox
        x = (col + obj_abox[C + 1]) * cell_size
        y = (row + obj_abox[C + 2]) * cell_size
        w = obj_abox[C + 3] * target_size
        h = obj_abox[C + 4] * target_size
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # class
        cls = int(np.argmax(obj_abox[:C]))
        cv2.putText(img, CLASSES[cls], (x1, y1), font, 0.5, (0,0,255), 1)
    cv2.imwrite(img_out, img)

if __name__ == '__main__':
    '''
    labels = format_label('../KITTI/training/split_0.1/train/label', 7, 2, 448, (float(448) / 1242, float(448) / 375))
    key = sys.argv[1]
    img_path_in = '../KITTI/training/split_0.1/train/image/0/%s.png' % key
    img_path_out = './%s.png' % key
    display(img_path_in, img_path_out, labels[key])
    '''

    img_in = sys.argv[1]
    img_out = sys.argv[2]
    yolo = Yolo()
    yolo.load_weights('./weights_coco.h5')
    y = yolo.predict(img_in)
    display(img_in, img_out, y[0])
