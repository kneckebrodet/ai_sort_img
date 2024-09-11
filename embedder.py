from keras_facenet import FaceNet
import numpy as np

class Embedder:
    def __init__(self):
        self.embedder = FaceNet()

    def _single_face_embedding(self, face_image):
        face_image = face_image.astype("float32")
        face_image = np.expand_dims(face_image, axis=0)
        yhat = self.embedder.embeddings(face_image)

        return yhat[0]

    def face_embeddings(self, face_images):
        EMBEDDED_FACES = []
        for image in face_images:
            EMBEDDED_FACES.append(self._single_face_embedding(image))

        return EMBEDDED_FACES
