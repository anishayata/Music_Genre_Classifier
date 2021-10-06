import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import visualkeras
from PIL import ImageFont

DATAPATH = "data_30.json"


def load_data(datapath):

    with open(datapath, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    Y = np.array(data["label"])
    classes = data["mapping"]

    return X, Y, classes


def prepare_dataset(test_size, validation_size):

    # load data
    X, Y, classes = load_data(DATAPATH)

    # create train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    # can't return values directly
    # for cnn, tf expects 3d array for each sample. the samples are actually 2d array of dim 130*13
    # 3d array -> (130, 13, 1) (how 130???)
    X_train = X_train[..., np.newaxis]  # gives a 4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test, classes


def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it to dense layer
    # 2d array to 1d array
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, Y):

    # X is a 3d array -> (130, 13, 1). predict() expects a 4d array ->(num_samples, 130, 13, 1)
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    # prediction is a 2d array ->[[0.1, 0.2 ...(10 values for 10 genre)]]
    predicted_index = np.argmax(prediction, axis=1) # [3]
    print("Expected index:{} and Predicted index: {}".format(Y, predicted_index))


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()



def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")

    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":

    # create train, validation and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test, classes = prepare_dataset(0.25, 0.2)

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the net
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # train CNN
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=32, epochs=45)

    # evaluate the CNN on test set
    test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    plot_history(history)

    # make predictions on a sample
    '''X = X_test[105]
    Y = Y_test[105]
    # print(Y)
    predict(model, X, Y)'''

    model.save('cnn_model.h5')
    summary = model.summary()

    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(Y_test, y_pred)
    plot_confusion_matrix(cm, classes, title='Confusion Matrix')

    font = ImageFont.truetype("arial.ttf", 32)
    visualkeras.layered_view(model).show()  # display using your system viewer
    visualkeras.layered_view(model, to_file='cnn.png', legend=True, font=font)  # write to disk
    visualkeras.layered_view(model, to_file='cnn.png', legend=True, font=font).show()  # write and show





