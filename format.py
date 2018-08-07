import sys
import numpy as np
import os

classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
anchor_boxes = [[60, 100], [100, 60]]
C = len(classes)

def read_kitti_label(fpath):
    objs = []
    with open(fpath, 'r') as fin:
        for line in fin:
            cols = line.split(' ')
            if cols[0] == 'DontCare':
                continue
            objs.append({
                'class': classes.index(cols[0]),
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


def format_label(path, S, B, target_size, resize=(1, 1)):
    label_dict = {}
    unit_size = target_size / S
    for fname in os.listdir(path):
        key = fname.split('.')[0]
        fpath = os.path.join(path, fname)
        objs = read_kitti_label(fpath)
        #label = np.zeros((S * S, 5 * B + C))
        label = np.zeros((S * S, B, 5 + C))
        for obj in objs:
            left = obj['bbox'][0] * resize[0]
            top = obj['bbox'][1] * resize[1]
            right = obj['bbox'][2] * resize[0]
            bottom = obj['bbox'][3] * resize[1]
            x = (right + left) / 2
            y = (bottom + top) / 2
            w = right - left
            h = bottom - top
            row = int(y // unit_size)
            col = int(x // unit_size)
            cls = obj['class']
            # TODO: determined bbox based on IOU
            x_unit = (col + 0.5) * unit_size
            y_unit = (row + 0.5) * unit_size
            IOU1 = cal_IOU([x, y, w, h], [x_unit - anchor_boxes[0][0] / 2, y_unit - anchor_boxes[0][1] / 2, x_unit + anchor_boxes[0][0] / 2, y_unit + anchor_boxes[0][1] / 2])
            IOU2 = cal_IOU([x, y, w, h], [x_unit - anchor_boxes[1][0] / 2, y_unit - anchor_boxes[1][1] / 2, x_unit + anchor_boxes[1][0] / 2, y_unit + anchor_boxes[1][1] / 2])
            # [C_class acnhor1_has_obj acnhor2_has_obj bbox1... bbox2... ]
            anc_box = 0
            if IOU2 > IOU1:
                anc_box = 1
            label[row * S + col][anc_box][cls] = 1
            label[row * S + col][anc_box][C] = 1
            label[row * S + col][anc_box][C + 1] = (x - col * unit_size) / unit_size
            label[row * S + col][anc_box][C + 2] = (y - row * unit_size) / unit_size
            label[row * S + col][anc_box][C + 3] = w / target_size
            label[row * S + col][anc_box][C + 4] = h / target_size
            if fname == '001137.txt':
                print 'format row: {0} col: {1}'.format(row, col)
                print label[row * S + col][anc_box]
                #print label[row * S + col][anc_box][C + 1] * unit_size + col * unit_size - w / 2
                print label[row * S + col][anc_box][C + 1] * unit_size + col * unit_size
            #print label[row * S + col]
        # label = label.flatten()

        label_dict[key] = label
    return label_dict

if __name__ == '__main__':
    # original size of KITTI is 1242 x 375
    labels = format_label(sys.argv[1], 7, 2, 448, (float(448) / 1242, float(448) / 375))
    import pickle
    with open(sys.argv[2], 'wb') as fout:
        pickle.dump(labels, fout)
