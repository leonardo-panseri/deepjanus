import tf_keras as keras


def load_data(num_classes):
    """Loads data from the MNIST dataset and preprocesses it for training."""
    # The data, 4 numpy arrays split between train and test sets
    # x_train: train data 60k
    # y_train: train label
    # x_test: test data 10k
    # y_test: test label
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Input image dimensions, images are 28x28 numpy arrays with int pixel values ranging from 0 to 255
    img_rows, img_cols = 28, 28

    # To be able to use the dataset in Keras API, we need 4-dims numpy arrays
    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape_bitmap_as_model_input(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape_bitmap_as_model_input(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape_bitmap_as_model_input(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape_bitmap_as_model_input(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255.
    x_test /= 255.

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape


def create_model(input_shape, num_classes, low_quality=False):
    """Builds and compiles the Keras CNN model for handwritten digit classification."""
    # MODEL CONFIGURATION
    # Add layers to the model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    # The following layer has 128 neurons
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    # Last layer returns an array of 10 probability scores that sum to 1.
    # Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
    # A softmax layer outputs a probability distribution, which means that each of the numbers can be interpreted as
    # a probability (in the range 0-1) representing the likelihood that the input pattern is an example of the
    # corresponding classification category.
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # MODEL COMPILATION
    # To compile the model, we have to define some settings
    # Loss function: measures how accurate the model is during training. We want to minimize this function.
    # Optimizer: how the model is updated based on the data it sees and its loss function.
    # Metrics: monitor the training and testing steps. We use accuracy, the fraction of the images that are correctly classified.
    if low_quality:
        optimizer = keras.optimizers.SGD(lr=0.001)
    else:
        optimizer = keras.optimizers.Adadelta()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def train_model(save_file_path='./data/models/cnnClassifier_trained', batch_size=128, epochs=12):
    """Trains the CNN model for handwritten digit classification on the MNIST dataset."""
    num_classes = 10
    (x_train, y_train), (x_test, y_test), input_shape = load_data(num_classes)
    model = create_model(input_shape, num_classes)

    # MODEL TRAINING
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # In our experiments:
    # gets to 99.25 % test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    # 16 seconds per epoch on a GRID K520 GPU.

    # MODEL EVALUATION
    # Compare how the trained model performs on the test dataset
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # EXPORT MODEL ARCHITECTURE AND WEIGHTS
    # Exporting the entire model allows to use the model for inference without access to the original code.
    model.save(save_file_path)
