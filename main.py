from gui import Menu
import os, shutil
from train import ScanAndExtractImage
from embedder import Embedder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import numpy as np


FOLDERS = {"models":"models", "output":"output"}

class main():
    ## Make sure the necessarry folders are created and in place
    for folder in FOLDERS:
        if not os.path.exists(folder):
                os.makedirs(folder)

    embedder = Embedder()
    image_handler = ScanAndExtractImage()

    npz_files = [file.split("_")[0] for file in os.listdir(FOLDERS["models"]) if file.endswith(".pkl")]
    menu = Menu(npz_files)
    target_inputs = menu.run()  ## ('target name',[list of training images IF NEW target], 'folder path')

    ## New target
    if len(target_inputs) > 2:
        X, y = image_handler.create_target(target_inputs[0], target_inputs[1])

        # Embedd image and save to file
        EMBEDDED_X = embedder.face_embeddings(X)
        EMBEDDED_X = np.asarray(EMBEDDED_X)
        # numerify targets from strings and put in order (name first for save training)
        encoded_y = []
        for target in y:
            if target == target_inputs[0]: #if target in y == name of target
                encoded_y.append(0)
            else:
                encoded_y.append(1)
        encoded_y.sort()
        print(y)
        print(encoded_y)

    else:
        print("Something went wrong with user inputs.")


    ## New target:
    if len(target_inputs) > 2:
        # train model and save
        X_train, X_test, y_train, y_test = train_test_split(EMBEDDED_X, encoded_y, test_size=0.3, random_state=1)
        model = SVC(kernel="linear", probability=True)
        model.fit(X_train, y_train)
        pickle.dump(model, open(f"{FOLDERS['models']}/{target_inputs[0]}_model.pkl", "wb"))
        model = pickle.load(open(f"{FOLDERS['models']}/{target_inputs[0]}_model.pkl", "rb"))
    else:
        model = pickle.load(open(f"{FOLDERS['models']}/{target_inputs[0]}_model.pkl", "rb"))


    ## check image if match or not (0 = match, 1 = no match)
    person_images, image_paths = image_handler.load_faces(target_inputs[-1]+"/")

    for index, person in enumerate(person_images):
        possible_target = embedder.face_embeddings([person])
        is_match = model.predict(possible_target)

        if is_match == 0:
            shutil.copy2(target_inputs[-1]+"/"+image_paths[index], FOLDERS["output"])

main()