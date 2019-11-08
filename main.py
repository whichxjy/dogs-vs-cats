from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def create_model():
    # Init Model
    model = Sequential()

    # [Block 1]
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    # [Dropout]
    model.add(Dropout(0.1))

    # [Block 2]
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # [Dropout]
    model.add(Dropout(0.1))

    # [Block 3]
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # [Dropout]
    model.add(Dropout(0.1))

    # [Flatten]
    model.add(Flatten())

    # [FC]
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # [Dropout]
    model.add(Dropout(0.3))

    # [Prediction]
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # create model
    model = create_model()
    # data generator for train
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # data generator for test
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory('dataset/train/',
        class_mode='binary', batch_size=64, target_size=(224, 224))
    test_it = test_datagen.flow_from_directory('dataset/test/',
        class_mode='binary', batch_size=64, target_size=(224, 224))
    # fit model
    model.fit_generator(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=1)
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('Classification Accuracy: %.3f' % (acc * 100.0))

if __name__ == '__main__':
    main()