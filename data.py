from keras import backend
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator, Iterator
import numpy as np
from six.moves import range
import os
import sys
from format import format_label

class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, labels, target_size=(256, 256),
                 color_mode='rgb', classes=None, class_mode='categorical', batch_size=32,
                 shuffle=True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix='', save_format='jpeg', follow_links=False):
        super(DirectoryIteratorWithBoundingBoxes, self).__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.labels = labels
        self.image_shape = target_size + (3,)

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            #index_array, current_index, current_batch_size = next(self.index_generator)
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((self.batch_size,) + self.image_shape, dtype=backend.floatx())
        label_dim = self.labels.values()[0].shape
        batch_y = np.zeros((self.batch_size, ) + label_dim)
        #locations = np.zeros((current_batch_size,) + (4,), dtype=backend.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            fkey = self.filenames[j].split('/')[-1].split('.')[0]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x /= 255
            #x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.labels[fkey]
            """
            if self.labels is not None:
                labels = self.labels[fname]
                locations[i] = np.asarray(
                        [labels.origin.x, labels.origin.y, labels.width, labels.height],
                        dtype=backend.floatx())
            """
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(batch_x.shape[0]):
                img = image.array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        """
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        if self.labels is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y
        """
        return batch_x, batch_y


if __name__ == "__main__":
    imgs_path  = sys.argv[1]
    label_path = sys.argv[2]
    labels = format_label(label_path, 7, 2, 448, (float(448) / 1242, float(448) / 375))
    iterator = DirectoryIteratorWithBoundingBoxes(imgs_path, ImageDataGenerator(), labels, target_size=(448, 448),
                                                  batch_size=16)
    batch_x, batch_y = next(iterator)
    print batch_x.shape, batch_y.shape
    print batch_x[0]
    # batch_y.shape
