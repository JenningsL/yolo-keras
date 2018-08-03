from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU,  MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from data import DirectoryIteratorWithBoundingBoxes
from format import format_label
import os
import math
import numpy as np

IMAGE_H = 448
IMAGE_W = 448

def yolo_loss(y_true, y_pred):
    coord_weight = 5
    noobj_weight = 0.5
    S = 7
    B = 2
    C = 9
    err = 0
    for unit_i in range(S * S):
        # [C_class acnhor1_has_obj acnhor2_has_obj bbox1... bbox2... ]
        for anchor_i in range(B):
            has_obj = int(y_true[unit_i][C + anchor_i])
            no_obj = int(1 - y_true[unit_i][C + anchor_i])
            coord_x_pred = y_pred[C + B + anchor_i * 4]
            coord_y_pred = y_pred[C + B + anchor_i * 4 + 1]
            coord_x_true = y_true[C + B + anchor_i * 4]
            coord_y_true = y_true[C + B + anchor_i * 4 + 1]
            # bbox coord error
            err += coord_weight * has_obj * (math.pow(coord_x_pred - coord_x_true, 2) + math.pow(coord_y_pred - coord_y_true, 2))
            # bbox size error
            w_pred = y_pred[C + B + anchor_i * 4 + 2]
            h_pred = y_pred[C + B + anchor_i * 4 + 3]
            w_true = y_true[C + B + anchor_i * 4 + 2]
            h_true = y_true[C + B + anchor_i * 4 + 3]
            err += has_obj * (math.pow(math.sqrt(w_pred) - math.sqrt(w_true), 2) + math.pow(math.sqrt(h_pred) - math.sqrt(h_true), 2))
            # obj confidence error
            err += math.pow(y_true[unit_i][C + anchor_i] - y_pred[unit_i][C + anchor_i], 2)
            err += no_obj * math.pow(y_true[unit_i][C + anchor_i] - y_pred[unit_i][C + anchor_i], 2)
        # classification error
        cls_pred = y_pred[:C]
        cls_true = y_true[:C]
        err += np.sum((cls_pred - cls_true)**2, axis=1)
    return err

def yolo_batch_loss(y_true, y_pred):
    batch_err = np.zeros((y_true.shape[0],))
    for i in y_true.shape[0]:
        batch_err[i] = yolo_loss(y_true[i], y_pred[i])
    return K.mean(batch_err)

class Yolo(object):
    """docstring for Yolo."""
    def __init__(self):
        super(Yolo, self).__init__()
        self.S = 7 # grid cell num in each axis
        self.B = 2 #
        self.C = 9 # number of classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.model = self._build()

    def _build(self):
        model = Sequential()

        #input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
        #output  = Input(shape=(self.S*self.S*(self.C+5*self.B)))
        model.add(Conv2D(64, 7, 2, input_shape=(448, 448, 3), activation=LeakyReLU(alpha=0.1)))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(192, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(128, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(256, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(256, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(512, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(256, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(512, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(256, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(512, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(256, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(512, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(256, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(512, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(512, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(1024, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(512, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(1024, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(512, 1, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(1024, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(1024, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(1024, 3, 2, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(1024, 3, 1, activation=LeakyReLU(alpha=0.1)))
        model.add(Conv2D(1024, 3, 1, activation=LeakyReLU(alpha=0.1)))

        model.add(Flatten())
        model.add(Dense(units=512, activation=LeakyReLU(alpha=0.1)))
        model.add(Dense(units=4096, activation=LeakyReLU(alpha=0.1)))
        model.add(Dense(units=self.S * self.S * (self.C + 5 * self.B)))
        model.add(Reshape(self.S * self.S, self.C + 5 * self.B))

        model.compile(loss=self.yolo_loss,
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def yolo_loss(self, y_true, y_pred):
        # [[
        #   [C_class acnhor1_has_obj acnhor2_has_obj bbox1... bbox2... ]
        # ]]
        # compute loss bbox
        loss_bbox = K.zeros(shape=(int_shape(y_true)[0], int_shape(y_true)[1], self.B * 4))
        for i in range(self.B):
            loss_bbox += K.square(y_true[:,:,(self.C + self.B + i * 4):(self.C + self.B + i * 4 + 2)] - y_pred[:,:,(self.C + self.B + i * 4):(self.C + self.B + i * 4 + 2)])
            loss_bbox += K.square(K.sqrt(y_true[:,:,(self.C + self.B + i * 4 + 2):(self.C + self.B + i * 4 + 4)]) - K.sqrt(y_pred[:,:,(self.C + self.B + i * 4 + 2):(self.C + self.B + i * 4 + 4)]))
        loss_bbox = K.sum(loss_bbox, axis=1) * self.lambda_coord
        # compute loss class
        loss_class = K.sum(K.square( y_pred[:,:,:self.C]  - y_true[:,:,:self.C]), axis=2)
        loss_class = K.sum(loss_class, axis=1)
        # TODO:
        loss_confidence = K.zeros(shape=int_shape(y_true))
        return K.mean(loss_bbox) + K.mean(loss_confidence) + K.mean(loss_class)

    def loss_yolo(self, y_true, y_pred):
        '''
            dense layer : Sx * Sy * B * ((5) + C)
            bbox : Sx * Sy * B * 4
            confidence = Sx * Sy * B * 1
            class : Sx * Sy * B * C

            7*7*4 = 196
            7*7*1 = 49
            7*7*20 = 980
        '''
        # reshape into cell
        y_true_bbox = sigmoid(K.reshape(y_true[:, :self.Sx*self.Sy*4*self.B], (-1, self.Sy, self.Sx, self.B, 4)))
        y_pred_bbox = sigmoid(K.reshape(y_pred[:, :self.Sx*self.Sy*4*self.B], (-1, self.Sy, self.Sx, self.B, 4)))
        y_true_confidence = sigmoid(K.reshape(y_true[:, self.Sx*self.Sy*4*self.B:self.Sx*self.Sy*5*self.B], (-1, self.Sy, self.Sx, self.B)))
        y_pred_confidence = sigmoid(K.reshape(y_pred[:, self.Sx*self.Sy*4*self.B:self.Sx*self.Sy*5*self.B], (-1, self.Sy, self.Sx, self.B)))
        y_true_class = softmax(K.reshape(y_true[:, self.Sx*self.Sy*5*self.B:], (-1, self.Sy, self.Sx, self.C)), axis=3)
        y_pred_class = softmax(K.reshape(y_pred[:, self.Sx*self.Sy*5*self.B:], (-1, self.Sy, self.Sx, self.C)), axis=3)

        # keep only boxes which exist in the dataset, if not put 0
        y_pred_bbox = y_pred_bbox * K.cast((y_true_bbox > 0), dtype='float32')

        # compute loss bbox
        loss_bbox = K.reshape(K.square( y_true_bbox[:,:,:,:,0:2] - y_pred_bbox[:,:,:,:,0:2]), (-1, self.Sx*self.Sy*2*self.B)) + K.reshape(K.square( K.sqrt(y_true_bbox[:, :, :, :, 2:]) - K.sqrt(y_pred_bbox[:, :, :, :, 2:])), (-1, self.Sx*self.Sy*2*self.B))
        loss_bbox = K.sum(loss_bbox, axis=1)*self.lambda_coord

        # compute loss confidence
        xmin_true = y_true_bbox[:,:,:,:, 0] - y_true_bbox[:,:,:,:, 2]/2
        ymin_true = y_true_bbox[:,:,:,:, 1] - y_true_bbox[:,:,:,:, 3]/2
        xmax_true = y_true_bbox[:,:,:,:, 0] + y_true_bbox[:,:,:,:, 2]/2
        ymax_true = y_true_bbox[:,:,:,:, 1] + y_true_bbox[:,:,:,:, 3]/2

        xmin_pred = y_pred_bbox[:,:,:,:, 0] - y_pred_bbox[:,:,:,:, 2]/2
        ymin_pred = y_pred_bbox[:,:,:,:, 1] - y_pred_bbox[:,:,:,:, 3]/2
        xmax_pred = y_pred_bbox[:,:,:,:, 0] + y_pred_bbox[:,:,:,:, 2]/2
        ymax_pred = y_pred_bbox[:,:,:,:, 1] + y_pred_bbox[:,:,:,:, 3]/2

        #print(' Xmin true : ', K.int_shape(xmin_true))

        xA = K.maximum(xmin_true, xmin_pred)
        yA = K.maximum(ymin_true, ymin_pred)
        xB = K.minimum(xmax_true, xmax_pred)
        yB = K.minimum(ymax_true, ymax_pred)
        #print('Xa : ', K.int_shape(xA))
        #if xA < xB and yA < yB:
        #condition1 = K.cast((xA<xB), dtype='float32')
        #condition2 =  K.cast( (yA<yB), dtype='float32')
        #condition = condition1 + condition2
        condition = K.cast((xA<xB), dtype='float32') + K.cast( (yA<yB), dtype='float32')
        # find which iou to compute
        tocompute = K.cast( K.equal(condition, 2.0), dtype='float32')
        del condition
            # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA) * tocompute
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
        boxBArea = (xmax_true - xmin_true) * (ymax_true - ymin_true)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        eps=0.0000001
        iou = (interArea / (eps+boxAArea + boxBArea - interArea)) * y_true_confidence * y_pred_confidence
        #print('iou shape : ', K.int_shape(iou))
        #print('tocompute shape : ', K.int_shape(tocompute))
        conf_obj = iou - y_pred_confidence*y_true_confidence
        conf_nobj = y_pred_confidence * K.cast( (y_true_confidence<1.0), dtype='float32')
        loss_confidence = K.reshape( K.square(conf_obj), (-1, self.Sy*self.Sx*self.B)) + self.lambda_noobj*K.reshape( K.square(conf_nobj), (-1, self.Sy*self.Sx*self.B))
        loss_confidence = K.sum(loss_confidence, axis=1)
        #print('loss confidence shape :', K.int_shape(loss_confidence))


        # keep only prediction class if there is an object in the cell, else put class to 0
        y_pred_class = (K.reshape(y_true_confidence[:,:,:,0], (-1, self.Sy, self.Sx, 1)) * y_pred_class)

        # compute loss class
        loss_class = K.sum(K.square( y_pred_class  - y_true_class), axis=3)
        loss_class = K.sum(loss_class, axis=2)
        loss_class = K.sum(loss_class, axis=1)
        #print(K.int_shape(loss_bbox))
        #print(K.int_shape(loss_class))

        loss = K.mean(loss_bbox) + K.mean(loss_confidence) + K.mean(loss_class)
        #loss = K.mean(loss_confidence) + K.mean(loss_class) #K.mean(loss_bbox) #K.mean(loss_confidence) #K.mean(self.lambda_noobj * loss_confidence)
        return loss

    def train(self, train_path, valid_path):
        train_labels = format_label(os.path.join(train_path, 'label'), 7, 2, 448, (448 / 1242, 448 / 375))
        train_generator = DirectoryIteratorWithBoundingBoxes(
                os.path.join(train_path, 'image'), ImageDataGenerator(), train_labels, target_size=(448, 448),
                batch_size=16)
        #batch_x, batch_y = train_generator.next()
        #print batch_x.shape, batch_y.shape
        validation_labels = format_label(os.path.join(valid_path, 'label'), 7, 2, 448, (448 / 1242, 448 / 375))
        validation_generator = DirectoryIteratorWithBoundingBoxes(
                os.path.join(valid_path, 'image'), ImageDataGenerator(), validation_labels, target_size=(448, 448),
                batch_size=16)
        self.model.fit_generator(
            train_generator,
            # steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=100)
        loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=16)
        print loss_and_metrics

#classes = model.predict(x_test, batch_size=128)

if __name__ == '__main__':
    yolo = Yolo()
    yolo.train('../KITTI/training/split_0.1/train', '../KITTI/training/split_0.1/valid')
