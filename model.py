from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU,  MaxPooling2D, Flatten, Reshape
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
        #model.add(Dense(units=self.S * self.S * (self.C + 5 * self.B)))
        #model.add(Reshape((self.S * self.S, self.C + 5 * self.B)))
        model.add(Dense(units=self.S * self.S * self.B * (self.C + 5)))
        model.add(Reshape((self.S * self.S, self.B, self.C + 5)))

        model.compile(loss=self.yolo_loss,
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def yolo_loss(self, y_true, y_pred):
        # [[
        #   [C_class acnhor1_has_obj acnhor2_has_obj bbox1... bbox2... ]
        # ]]
        # compute loss bbox
        loss_bbox = K.square(y_true[..., 1:3] - y_pred[..., 1:3]) + K.square(K.sqrt(y_true[..., 3:5]) - K.sqrt(y_pred[..., 3:5]))
        loss_bbox = K.sum(loss_bbox, axis=2)
        loss_bbox = K.sum(loss_bbox, axis=1) * self.lambda_coord
        # compute loss class
        loss_class = K.sum(K.square( y_pred[...,:self.C]  - y_true[...,:self.C]), axis=2)
        loss_class = K.sum(loss_class, axis=1)
        # TODO:
        loss_confidence = K.sum(K.square(y_true[..., 0] - y_pred[..., 0]), axis=2)
        loss_confidence = K.sum(loss_confidence, axis=1)
        return K.mean(loss_bbox) + K.mean(loss_confidence) + K.mean(loss_class)

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
