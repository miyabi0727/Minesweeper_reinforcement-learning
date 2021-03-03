from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
# from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os

folder = os.listdir("cell")
# folder.pop(-1)
image_size = 30
dense_size  = len(folder)

X = []
Y = []
# for index, name in enumerate(folder):
dense_size = 2
for index, name in enumerate(['cell_open','cell_close']):
    dir = "./cell/" + name
    files = glob.glob(dir + "/*.png")
    print(name)
    for i, file_path in enumerate(files):
        image = Image.open(file_path)
        image = image.convert("RGB")
        # image = image.resize((image_size, image_size)
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)
X = X.astype('float32')
X = X / 255.0

Y = np_utils.to_categorical(Y, dense_size)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1)

# モデルの定義
model = Sequential()
model.add(Conv2D(5, (3, 3), padding='same',input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(5, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(25, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(25, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(dense_size))
model.add(Activation('softmax'))

model.summary()

optimizers = Adam(learning_rate=0.0001)
results = {}
print(Y)


epochs = 2000
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
results[0] = model.fit(X, Y, batch_size=2, epochs=epochs )

model_json_str = model.to_json()
open('model_v2.json', 'w').write(model_json_str)
model.save('weights_open.h5')


x = range(epochs)
for k, result in results.items():
    plt.plot(x, result.history['accuracy'], label=k)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

name = 'acc.jpg'
plt.savefig(name, bbox_inches='tight')
plt.close()

# for k, result in results.items():
#     print(result.history)
#     plt.plot(x, result.history['val_accuracy'], label=k)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

# name = 'val_acc.jpg'
# plt.savefig(name, bbox_inches='tight')