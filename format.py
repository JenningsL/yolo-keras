import sys
import numpy as np
import os
from config import C, S, B, batch_size, origin_size, target_size, CLASSES, anchor_boxes

# anchor_boxes = [[60, 100], [100, 60]]
cell_size = target_size[0] / S

def read_kitti_label(fpath):
    objs = []
    with open(fpath, 'r') as fin:
        for line in fin:
            cols = line.split(' ')
            if cols[0] == 'DontCare':
                continue
            objs.append({
                'class': CLASSES.index(cols[0]),
                'bbox': [float(cols[4]), float(cols[5]), float(cols[6]), float(cols[7])] # left, top, right, bottom
            })
    return objs

def cal_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def find_anchor_box(true_box, col, row):
    grid_center_x = (col + 0.5) * cell_size
    grid_center_y = (row + 0.5) * cell_size
    IOUs = []
    for abox in anchor_boxes:
        box_w = cell_size * abox[0]
        box_h = cell_size * abox[1]
        anchor_box = [grid_center_x - box_w / 2, grid_center_y - box_h / 2, grid_center_x + box_w / 2, grid_center_y + box_h / 2]
        iou = cal_IOU(true_box, anchor_box)
        IOUs.append(iou)
    return np.argmax(IOUs)

def format_label(path):
    resize = (float(target_size[0]) / origin_size[0], float(target_size[1]) / origin_size[1])
    label_dict = {}
    for fname in os.listdir(path):
        key = fname.split('.')[0]
        fpath = os.path.join(path, fname)
        objs = read_kitti_label(fpath)
        #label = np.zeros((S * S, 5 * B + C))
        label = np.zeros((S, S, B, 5 + C))
        for obj in objs:
            left = obj['bbox'][0] * resize[0]
            top = obj['bbox'][1] * resize[1]
            right = obj['bbox'][2] * resize[0]
            bottom = obj['bbox'][3] * resize[1]
            x = (right + left) / 2
            y = (bottom + top) / 2
            w = right - left
            h = bottom - top
            row = int(y // cell_size)
            col = int(x // cell_size)
            cls = obj['class']
            # TODO: determined bbox based on IOU
            anc_box = find_anchor_box([left, top, right, bottom], col, row)
            label[row][col][anc_box][cls] = 1
            label[row][col][anc_box][C] = 1
            label[row][col][anc_box][C + 1] = (x - col * cell_size) / cell_size
            label[row][col][anc_box][C + 2] = (y - row * cell_size) / cell_size
            label[row][col][anc_box][C + 3] = w / target_size[0]
            label[row][col][anc_box][C + 4] = h / target_size[0]
            #print label[row * S + col]
        # label = label.flatten()

        label_dict[key] = label
    return label_dict

if __name__ == '__main__':
    # original size of KITTI is 1242 x 375
    labels = format_label(sys.argv[1])
    import pickle
    with open(sys.argv[2], 'wb') as fout:
        pickle.dump(labels, fout)
