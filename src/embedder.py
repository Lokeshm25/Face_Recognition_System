import numpy as np
from keras_facenet import FaceNet


class FaceEmbedder:
    def __init__(self):
        self.embedder = FaceNet()

    def get_embedding(self, face_img):
        """
        face_img: RGB uint8 160x160x3
        returns: 512-d numpy array
        """
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = self.embedder.embeddings(face_img)
        return yhat[0]
