import os, cv2, shutil
import numpy as np
from mtcnn.mtcnn import MTCNN


class ScanAndExtractImage:
    def __init__(self):
        self.directory = "train"
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]["box"]
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv2.resize(face, self.target_size)
        return face_arr

    def load_faces(self, directory):
        faces = []
        paths = []
        for image_name in os.listdir(directory):
            try:
                path = directory+image_name
                single_face = self.extract_face(path)
                faces.append(single_face)
                paths.append(image_name)
            except Exception as e:
                pass

        return faces, paths

    def create_target(self, name, images):
        # check if directory "name" exists, else create dir
        path = self.directory+"/"+name+"/"
        if not os.path.exists(path):
            os.makedirs(path)

        # copy training images to "name" directory
        for image in images:
            shutil.copy2(image, path)

        # get training pics of target
        for image in path:
            target_faces, _ = self.load_faces(path)
            target_labels = [name for _ in range(len(target_faces))]

        # loop through folders in directory
        for folder_name in os.listdir(self.directory):
            # grab extracted faces from all images
            if folder_name == name:
                continue
            else:
                other_faces, _ = self.load_faces(self.directory +"/"+ folder_name + "/")
                other_labels = ["other" for _ in range(len(other_faces))]

        faces = target_faces + other_faces
        labels = target_labels + other_labels

        self.X.extend(faces)
        self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)



