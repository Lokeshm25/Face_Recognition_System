import numpy as np
from keras_facenet import FaceNet

class FaceEmbedder:
    """
    Wrapper around keras-facenet FaceNet.
    Converts 160x160x3 RGB face images into 512-d embeddings.
    """

    def __init__(self):
        self.embedder = FaceNet()

    def get_embedding(self, face_img):
        """
        face_img: RGB uint8 image of shape (160, 160, 3)
        returns: 1D numpy array of length 512
        """
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = self.embedder.embeddings(face_img)
        return yhat[0]
