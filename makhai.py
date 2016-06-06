import dataio
import numpy as np

def train():
    from keras.callbacks import ModelCheckpoint
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD
    from keras.utils import np_utils

    (X_train, y_train), (X_test, y_test) = dataio.load_dataset()
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 96, 128)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    structure = open("model/structure.txt", "w")
    structure.write(model.to_json())
    structure.close()

    model_checkpoint = ModelCheckpoint("model/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.txt")
    model.fit(X_train, y_train,
              batch_size=32,
              nb_epoch = 10,
              callbacks=[model_checkpoint],
              validation_data=(X_test, y_test))
    return model

def test(model):
    X_test, files = dataio.load_testset()
    X_test = np.concatenate([X_test],axis=0)
    y_out = model.predict_proba(X_test, batch_size=128, verbose=1)
    fout = open('model/results.txt', 'w')
    for i in range(len(files)):
        label = np.argmax(y_out[i])
        fout.write(str(files[i]) + " " + str(label) + "\n")
    fout.close()

if __name__ == "__main__":
    model = train()
    test(model)
