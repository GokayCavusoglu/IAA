import os
from PIL import Image
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

def cleanup_duplicates(path1, path2):
    files1 = set(os.listdir(path1))
    files2 = set(os.listdir(path2))
    
    common_files = files1.intersection(files2)
    for filename in common_files:
        file1 = os.path.join(path1, filename)
        file2 = os.path.join(path2, filename)
        if os.path.isfile(file1):
            os.remove(file1)
            print(f"Doublon supprimé : {file1}")
        if os.path.isfile(file2):
            os.remove(file2)
            print(f"Doublon supprimé : {file2}")

def buildSampleFromPath(path1, path2):
    h = 250
    l = 250
    S = []
    for filename in os.listdir(path1):
        file_path = os.path.join(path1, filename)

        img = Image.open(file_path).convert("RGB")
        img_resized = resizeImage(img, h, l)
        histo = computeHisto(img_resized)

        image_dict = {
            "name_path": file_path,
            "resized_image": img_resized,
            "X_histo": histo,
            "y_true_class": +1,
            "y_predicted_class": None
        }

        S.append(image_dict)
        img.close()

    for filename in os.listdir(path2):
        file_path = os.path.join(path2, filename)

        img = Image.open(file_path).convert("RGB")
        img_resized = resizeImage(img, h, l)
        histo = computeHisto(img_resized)

        image_dict = {
            "name_path": file_path,
            "resized_image": img_resized,
            "X_histo": histo,
            "y_true_class": -1,
            "y_predicted_class": None
        }

        S.append(image_dict)
        img.close()
    return S

def resizeImage(i, h, l):
    return i.resize((l, h))

def computeHisto(i):
    return i.histogram()

def fitFromHisto(S, algo):
    X = [img["X_histo"] for img in S]
    y = [img["y_true_class"] for img in S]
    if algo["name"] == "GaussianNB":
        model = GaussianNB(**algo["hyper_param"])
    else:
        raise ValueError("Algorithme pas reconnu")
    model.fit(X, y)
    return model

def predictFromHisto(S, model):
    predictions = []
    for img in S:
        x = img["X_histo"]
        y_pred = model.predict([x])[0]
        img["y_predicted_class"] = y_pred
        predictions.append(y_pred)
    return predictions

def erreurempirique(S):
    y_true = [img["y_true_class"] for img in S]
    y_pred = [img["y_predicted_class"] for img in S]

    accuracy = accuracy_score(y_true, y_pred)
    return 1 - accuracy


def crossValidationError(S, algo, k):
    X = [img["X_histo"] for img in S]
    y = [img["y_true_class"] for img in S]

    if algo["name"] == "GaussianNB":
        model = GaussianNB(**algo["hyper_param"])
    else:
        raise ValueError("Algorithme non reconnu")

    accuracies = cross_val_score(model, X, y, cv=k)
    return 1 - np.mean(accuracies)

path1 = r"C:/Users/Syssou/Downloads/Init_data/Init/Ailleurs" #remplacez le chemin par votre propre chemin ou pushez directement les 2 fichiers et remplacez ces 2 paths par les paths adéquats 
path2 = r"C:/Users/Syssou/Downloads/Init_data/Init/Mer"
cleanup_duplicates(path1, path2)
S = buildSampleFromPath(path1, path2)
print("Nombre d'images chargees :", len(S))
algo = {"name": "GaussianNB", "hyper_param": {}}
model = fitFromHisto(S, algo)
predictFromHisto(S, model)
print("Erreur empirique :", erreurempirique(S))
print("crossValidation:", crossValidationError(S, algo, k=5))