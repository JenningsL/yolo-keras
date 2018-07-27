from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU,  MaxPooling2D
from data import DirectoryIteratorWithBoundingBoxes
from format import format_label

class Yolo(object):
    """docstring for Yolo."""
    def __init__(self):
        super(Yolo, self).__init__()
        self.S = 7 # grid cell num in each axis
        self.B = 2 #
        self.C = 9 # number of classes

    def _build():
        model = Sequential()

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

        model.add(Dense(units=512, activation=LeakyReLU(alpha=0.1)))
        model.add(Dense(units=4096, activation=LeakyReLU(alpha=0.1)))
        model.add(Dense(units=self.S*self.S*(self.C+5*self.B)))

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(train_path, valid_path):
        train_labels = format_label(os.join(train_path, 'label'), 7, 2, 448, (448 / 1242, 448 / 375))
        train_generator = DirectoryIteratorWithBoundingBoxes(
                os.join(train_path, 'image'), ImageDataGenerator(), train_labels, target_size=(448, 448),
                batch_size=32)
        validation_labels = format_label(os.join(valid_path, 'label'), 7, 2, 448, (448 / 1242, 448 / 375))
        validation_generator = DirectoryIteratorWithBoundingBoxes(
                os.join(valid_path, 'image'), ImageDataGenerator(), validation_labels, target_size=(448, 448),
                batch_size=32)
        model.fit_generator(
            train_generator,
            # steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

#classes = model.predict(x_test, batch_size=128)

if __name__ == '__main__':
    model = Yolo()
    model.train('../KITTI/training', '../KITTI/valid')
