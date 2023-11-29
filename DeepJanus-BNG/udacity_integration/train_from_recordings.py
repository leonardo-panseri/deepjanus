import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
import os

from core.folders import FOLDERS
from udacity_integration.batch_generator import Generator
from udacity_integration.udacity_utils import INPUT_SHAPE

np.random.seed(0)


def load_data(args):
    """Loads training data and splits it into training and validation set."""
    tracks = [FOLDERS.training_recordings]

    x = np.empty([0, 3])
    y = np.array([])
    for track in tracks:
        drive = os.listdir(track)
        for drive_style in drive:
            csv_name = 'driving_log.csv'
            csv_folder = os.path.join(track, drive_style)
            csv_path = os.path.join(csv_folder, csv_name)
            try:
                def fix_path(series):
                    return series.apply(lambda d: os.path.join(csv_folder, d))

                data_df = pd.read_csv(csv_path)
                pictures = data_df[['center', 'left', 'right']]
                pictures_fixpath = pictures.apply(fix_path)
                csv_x = pictures_fixpath.values

                csv_y = data_df['steering'].values
                x = np.concatenate((x, csv_x), axis=0)
                y = np.concatenate((y, csv_y), axis=0)
            except FileNotFoundError:
                print("Unable to read file %s" % csv_path)
                exit()

    try:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    print("Train dataset: " + str(len(x_train)) + " elements")
    print("Test dataset: " + str(len(x_valid)) + " elements")
    return x_train, x_valid, y_train, y_valid


def build_model(args):
    """Builds the ML model for the lane-keeping assist system."""
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, x_train, x_valid, y_train, y_valid):
    """Trains the model."""
    os.makedirs('trained_models', exist_ok=True)
    checkpoint = ModelCheckpoint('trained_models/self-driving-car-{epoch:03d}-2020.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    train_generator = Generator(x_train, y_train, True, args)
    validation_generator = Generator(x_valid, y_valid, False, args)

    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        epochs=args.nb_epoch,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        workers=4,
                        callbacks=[checkpoint],
                        verbose=1)
    # use_validation_multiprocessing


def plot_dataset(y):
    plt.hist(y, 20, facecolor='blue', alpha=0.5)
    plt.show()


def main(args):
    """Loads train/validation data set and trains the model."""
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    plot_dataset(np.concatenate((data[2], data[3])))
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
