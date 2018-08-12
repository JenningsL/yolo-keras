from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU,  MaxPooling2D, Flatten, Reshape, BatchNormalization, Input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
from data import DirectoryIteratorWithBoundingBoxes
from format import format_label
import os
import math
import numpy as np

IMAGE_H = 448
IMAGE_W = 448
batch_size = 8

class Yolo(object):
    """docstring for Yolo."""
    def __init__(self):
        super(Yolo, self).__init__()
        self.S = 7 # grid cell num in each axis
        self.B = 2 #
        self.C = 9 # number of classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        #self.target_size = (448, 448)
        self.target_size = (416, 416)
        self.model = self._build()

    def _build(self):
        model = Sequential()

        #input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
        #output  = Input(shape=(self.S*self.S*(self.C+5*self.B)))
        model.add(Conv2D(64, 7, 2, input_shape=self.target_size + (3,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(192, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(128, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(256, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(512, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, 1, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, 3, 2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, 3, 1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(units=4096))
        model.add(LeakyReLU(alpha=0.1))
        #model.add(Dense(units=self.S * self.S * (self.C + 5 * self.B)))
        #model.add(Reshape((self.S * self.S, self.C + 5 * self.B)))

        model.add(Dense(units=self.S * self.S * self.B * (self.C + 5)))
        model.add
        model.add(Reshape((self.S * self.S, self.B, self.C + 5)))

        optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss=self.yolo_loss, optimizer=optimizer)
        return model

    def _build_v2():
        BOX = self.B
        CLASS = self.C
        GRID_H = self.S
        GRID_W = self.S

        input_image = Input(shape=self.target_size+(3,))
        # true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 23
        x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
        output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

        # small hack to allow true_boxes to be registered when Keras build the model
        # for more information: https://github.com/fchollet/keras/issues/2790
        # output = Lambda(lambda args: args[0])([output, true_boxes])

        # model = Model([input_image, true_boxes], output)
        self.model = Model(input_image, output)

    def load_pretrained_weights(self, wt_path):
        weight_reader = WeightReader(wt_path)
        weight_reader.reset()
        nb_conv = 23

        for i in range(1, nb_conv+1):
            conv_layer = self.model.get_layer('conv_' + str(i))

            if i < nb_conv:
                norm_layer = self.model.get_layer('norm_' + str(i))

                size = np.prod(norm_layer.get_weights()[0].shape)

                beta  = weight_reader.read_bytes(size)
                gamma = weight_reader.read_bytes(size)
                mean  = weight_reader.read_bytes(size)
                var   = weight_reader.read_bytes(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            if len(conv_layer.get_weights()) > 1:
                bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel])
        # Randomize weights of the last layer
        layer   = model.layers[-4] # the last convolutional layer
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
        new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

        layer.set_weights([new_kernel, new_bias])

    def yolo_loss(self, y_true, y_pred):
        y_pred = K.sigmoid(y_pred)
        coord_mask = K.expand_dims(y_true[..., self.C], axis=-1)
        confidence_mask = (K.ones_like(coord_mask) - coord_mask) * self.lambda_noobj + coord_mask
        # compute loss bbox
        loss_bbox = K.square(y_true[..., 1:3] - y_pred[..., 1:3]) + K.square(K.sqrt(y_true[..., 3:5]) - K.sqrt(y_pred[..., 3:5]))
        loss_bbox = loss_bbox * coord_mask
        loss_bbox = K.sum(loss_bbox, axis=3)
        loss_bbox = K.sum(loss_bbox, axis=2)
        loss_bbox = K.sum(loss_bbox, axis=1) * self.lambda_coord
        # compute loss class
        loss_class = K.sum(K.square(y_pred[...,:self.C]  - y_true[...,:self.C]) * coord_mask, axis=3)
        loss_class = K.sum(loss_class, axis=2)
        loss_class = K.sum(loss_class, axis=1)
        # TODO:
        loss_confidence = K.square(K.expand_dims(y_true[..., self.C] - y_pred[..., self.C], axis=-1)) * confidence_mask
        loss_confidence =  K.sum(loss_confidence, axis=3)
        loss_confidence = K.sum(loss_confidence, axis=2)
        loss_confidence = K.sum(loss_confidence, axis=1)
        #return K.mean(loss_bbox) + K.mean(loss_confidence) + K.mean(loss_class)
        #return loss_confidence
        return loss_bbox + loss_confidence + loss_class

    def train(self, train_path, valid_path):
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=3,
                                   mode='min',
                                   verbose=1)

        checkpoint = ModelCheckpoint('weights_coco.h5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=1)
        train_labels = format_label(os.path.join(train_path, 'label'), 7, 2, self.target_size[0], (float(self.target_size[0]) / 1242, float(self.target_size[1]) / 375))
        train_generator = DirectoryIteratorWithBoundingBoxes(
                os.path.join(train_path, 'image'), ImageDataGenerator(), train_labels, target_size=self.target_size,
                batch_size=batch_size)
        #batch_x, batch_y = train_generator.next()
        #print batch_x.shape, batch_y.shape
        validation_labels = format_label(os.path.join(valid_path, 'label'), 7, 2, self.target_size[0], (float(self.target_size[0]) / 1242, float(self.target_size[1]) / 375))
        validation_generator = DirectoryIteratorWithBoundingBoxes(
                os.path.join(valid_path, 'image'), ImageDataGenerator(), validation_labels, target_size=self.target_size,
                batch_size=batch_size)
        self.model.fit_generator(
            train_generator,
            # steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            callbacks=[early_stop, checkpoint],
            validation_steps=100)
        #loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=16)
        #print loss_and_metrics

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, fpath):
        img = image.load_img(fpath, target_size=self.target_size)
        x = image.img_to_array(img)
        #x /= 255
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x)

#classes = model.predict(x_test, batch_size=128)
def yolo_loss(y_true, y_pred):
    # [[
    #   [C_class acnhor1_has_obj acnhor2_has_obj bbox1... bbox2... ]
    # ]]
    C = 2
    lambda_coord = 5
    lambda_noobj = 0.5
    coord_mask = K.expand_dims(y_true[..., C], axis=-1)
    #coord_mask = y_true[..., C]
    confidence_mask = (K.ones_like(coord_mask) - coord_mask) * lambda_noobj + coord_mask
    # compute loss bbox
    loss_bbox = K.square(y_true[..., 1:3] - y_pred[..., 1:3]) + K.square(K.sqrt(y_true[..., 3:5]) - K.sqrt(y_pred[..., 3:5]))
    loss_bbox = loss_bbox * coord_mask
    loss_bbox = K.sum(loss_bbox, axis=3)
    loss_bbox = K.sum(loss_bbox, axis=2)
    loss_bbox = K.sum(loss_bbox, axis=1) * lambda_coord
    # compute loss class
    loss_class = K.sum(K.square(y_pred[...,:C]  - y_true[...,:C]) * coord_mask, axis=3)
    loss_class = K.sum(loss_class, axis=2)
    loss_class = K.sum(loss_class, axis=1)
    # TODO:
    loss_confidence = K.square(K.expand_dims(y_true[..., C] - y_pred[..., C], axis=-1)) * confidence_mask
    loss_confidence =  K.sum(loss_confidence, axis=3)
    loss_confidence = K.sum(loss_confidence, axis=2)
    loss_confidence = K.sum(loss_confidence, axis=1)
    #return K.mean(loss_bbox) + K.mean(loss_confidence) + K.mean(loss_class)
    #return loss_confidence
    return loss_bbox + loss_confidence + loss_class

if __name__ == '__main__':
    yolo = Yolo()
    yolo.load_weights('./yolo.weights')
    yolo.train('../KITTI/training/split_0.1/train', '../KITTI/training/split_0.1/valid')

    '''
    # test loss function
    y_true = Input(shape=(1, 2, 7))
    y_pred = Input(shape=(1, 2, 7))
    loss_func = K.Function([y_true, y_pred], [yolo_loss(y_true, y_pred)])
    print loss_func([
        [[[[1, 0, 1, 1, 0.1, 0.1, 0.1], [0, 0, 0, 0.6, 0.6, 0.6, 0.6]]]],
        [[[[0, 0, 1, 0.1, 0.1, 0.1, 0.1], [0, 1, 1, 0.5, 0.5, 0.5, 0.5]]]]
    ])
    '''
