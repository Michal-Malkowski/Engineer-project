import numpy as np
from skimage.color import rgb2gray

import tensorflow as tf
from keras.models import Model, load_model
from keras import backend as K
import cv2
import imagej
from skimage import io
from scyjava import jimport
import os

IMG_WIDTH = 512
IMG_HEIGHT = 512

class Processing():
    area = 0.0
    def __init__(self):
        self.ij = imagej.init('./Fiji.app')

    def mean_iou(y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def convert_gray2rgb(image):
        width, height = image.shape
        out = np.empty((width, height, 3), dtype=np.uint8)
        out[:, :, 0] = image
        out[:, :, 1] = image
        out[:, :, 2] = image
        return out

    def convert_gray2rgb2(image):
        out = np.empty((IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8)
        out[:, :, 0] = image[:, :, 0]
        out[:, :, 1] = image[:, :, 0]
        out[:, :, 2] = image[:, :, 0]
        return out

    def loadModel(self):
        self.model = load_model('./model/m.h5',
                                custom_objects={'mean_iou': self.mean_iou})

    def drawResize(self, frame):
        res = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        return res

    def drawSegmentation(self, frame, realTime):
        if realTime:
            imgRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resize = self.drawResize(self, imgRgb)
        else:
            img_resize = self.drawResize(self, frame)
        self.img_outline = img_resize
        newframe = np.expand_dims(img_resize, axis=0)
        preds = self.model.predict(newframe, verbose=0)
        preds_img = (preds > 0.5).astype(np.uint8)
        self.area = round(np.count_nonzero(preds_img == 1) * 100 / (IMG_WIDTH*IMG_HEIGHT), 2)
        g2r = self.convert_gray2rgb2(np.squeeze(preds_img * 255, axis=0))
        return g2r

    def getArea(self):
        return self.area

    def drawOutline(self, frame, realTime):
        imgSegmentation = self.drawSegmentation(self, frame, realTime)
        gray = rgb2gray(imgSegmentation)

        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        gray = cv2.cvtColor(imgSegmentation, cv2.COLOR_BGR2GRAY)
        contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            cv2.drawContours(mask, [c], -1, (36, 255, 12), thickness=3)

        mask = mask // 255
        mask3d = self.convert_gray2rgb(mask)

        for i in range(self.img_outline.shape[0]):
            for j in range(self.img_outline.shape[1]):
                if mask3d[i, j, 2] == 0:
                    self.img_outline[i, j, 2] = 255
                    self.img_outline[i, j, 0] = 0
                    self.img_outline[i, j, 1] = 0

        RGB_img = cv2.cvtColor(self.img_outline, cv2.COLOR_BGR2RGB)
        return RGB_img

    def drawMean(self, filename):
        self.ij = imagej.init('./Fiji.app')
        filename = filename.split("/")
        filename = filename[len(filename)-1]
        res = self.ij.io().open('../myDataSet/'+filename)
        HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
        radius = HyperSphereShape(1)
        result = self.ij.dataset().create(res)

        r = self.ij.op().filter().mean(result, res, radius)

        self.ij.io().save(r, 'temp.jpg')
        img = io.imread('./temp.jpg')

        os.remove('./temp.jpg')
        res = self.drawResize(self, img)
        return res

    def drawMedian(self, filename):
        self.ij = imagej.init('./Fiji.app')
        filename = filename.split("/")
        filename = filename[len(filename) - 1]
        res = self.ij.io().open('../myDataSet/' + filename)
        HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
        radius = HyperSphereShape(1)
        result = self.ij.dataset().create(res)

        r = self.ij.op().filter().median(result, res, radius)

        self.ij.io().save(r, 'temp.jpg')
        img = io.imread('./temp.jpg')

        os.remove('./temp.jpg')
        res = self.drawResize(self, img)
        return res

