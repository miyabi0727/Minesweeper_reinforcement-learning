from keras.models import load_model
import numpy as np
from PIL import Image
from glob import glob

class Cnn():
    def __init__(self):
        self.model = load_model('weights_v2.h5')
        self.labels = np.array(['cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell_close', 'cell_flag', 'cell_flag_miss', 'cell_open', 'cell_open_miss'])
        self.model_oc = load_model('weights_open.h5')

    def check_one(self, file_path):
        image = Image.open(file_path)
        image = image.convert("RGB")
        data = np.asarray(image)
        X = np.array([data])
        X = X.astype('float32')
        X = X / 255.0
        predict_classes = self.model.predict(X)
        class_index = [predict_classes[0].argmax()]
        
        if 6 in class_index or 9 in class_index:
            if data[15][15][1] > 245:
                class_index = [9]
            else:
                class_index = [6]
            # predict_classes = self.model_oc.predict(X)
            # class_index = [predict_classes[0].argmax()]
            # class_index = list([9 if 0 == class_index[0] else 6])
        return class_index
        
    def check_all(self, folder="capture/_*.png"):
        file_paths = glob(folder)
        # X = []
        class_index = []
        # for index, file in enumerate(file_paths):
        for file_path in file_paths:
            class_index.append(self.check_one(file_path)[0])
        #     image = Image.open(file)
        #     image = image.convert("RGB")
        #     data = np.asarray(image)
        #     X.append(data)
        #     # Y.append(index)
        # X = np.array(X)
        # # Y = np.array(Y)
        # X = X.astype('float32')
        # X = X / 255.0
        # predict_classes = self.model.predict(X)
        # return np.array([v.argmax() for v in predict_classes])
        return np.array(class_index)

    def check_55(self):
        class_index = []
        file_paths = [
            'capture/_08.png', 'capture/_09.png', 'capture/_10.png', 'capture/_11.png', 'capture/_12.png',
            'capture/_15.png', 'capture/_16.png', 'capture/_17.png', 'capture/_18.png', 'capture/_19.png', 
            'capture/_22.png', 'capture/_23.png', 'capture/_24.png', 'capture/_25.png', 'capture/_26.png',
            'capture/_29.png', 'capture/_30.png', 'capture/_31.png', 'capture/_32.png', 'capture/_33.png',
            'capture/_36.png', 'capture/_37.png', 'capture/_38.png', 'capture/_39.png', 'capture/_40.png'
            ]
        for file_path in file_paths:
            class_index.append(self.check_one(file_path)[0])
        return np.array(class_index)


    def check_labels(self, predict_classes):
        return [self.labels[class_num] for class_num in predict_classes]

    # def test_check(self, folder=)
if __name__ == '__main__':
    cnn = Cnn()
    # print(cnn.check_one('capture/_04.png'))
    # print(cnn.check_all(folder='cell/**/*.png'))
    print(cnn.check_all())
    print(cnn.check_55())