from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU,  MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from data import DirectoryIteratorWithBoundingBoxes
from format import format_label
import os

IMAGE_H = 448
IMAGE_W = 448
class Yolo(object):
    """docstring for Yolo."""
    def __init__(self):
        super(Yolo, self).__init__()
        self.S = 7 # grid cell num in each axis
        self.B = 2 #
        self.C = 9 # number of classes
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
        model.add(Dense(units=self.S*self.S*(self.C+5*self.B)))

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def train(self, train_path, valid_path):
        train_labels = format_label(os.path.join(train_path, 'label'), 7, 2, 448, (448 / 1242, 448 / 375))
        train_generator = DirectoryIteratorWithBoundingBoxes(
                os.path.join(train_path, 'image'), ImageDataGenerator(), train_labels, target_size=(448, 448),
                batch_size=32)
        #batch_x, batch_y = train_generator.next()
        #print batch_x.shape, batch_y.shape
        validation_labels = format_label(os.path.join(valid_path, 'label'), 7, 2, 448, (448 / 1242, 448 / 375))
        validation_generator = DirectoryIteratorWithBoundingBoxes(
                os.path.join(valid_path, 'image'), ImageDataGenerator(), validation_labels, target_size=(448, 448),
                batch_size=32)
        self.model.fit_generator(
            train_generator,
            # steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=100)
        loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=64)
        print loss_and_metrics

#classes = model.predict(x_test, batch_size=128)

if __name__ == '__main__':
    yolo = Yolo()
    yolo.train('../KITTI/training/split_0.1/train', '../KITTI/training/split_0.1/valid')
