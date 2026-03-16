#Fichier implémentant des algorithmes KNN , vector machine ....
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import joblib
from PIL import Image
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import shutil
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

def buildTestFromPath(path):
    h, l = 250, 250
    S_test = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception:
            continue
        img_resized = resizeImage(img, h, l)
        histo = computeHisto(img_resized)
        S_test.append({
            "name_path":          file_path,
            "resized_image":      img_resized,
            "X_histo":            histo,
            "y_true_class":       None,  
            "y_predicted_class":  None,
        })
        img.close()
    return S_test

def resizeImage(i, h, l):
    return i.resize((l, h))

def cropBottomThird(img):
    w, h = img.size
    return img.crop((0, 2 * h // 3, w, h))

def computeHisto(img):
    return cropBottomThird(img).histogram()

def fitFromHisto(S, algo):
    X = np.array([img["X_histo"] for img in S])
    y = [img["y_true_class"] for img in S]

    scaler = None
    if algo["name"] == "GaussianNB":
        model = GaussianNB(**algo["hyper_param"])
    elif algo["name"] == "RandomForest":
        model = RandomForestClassifier(**algo["hyper_param"])
    elif algo["name"] == "SVM":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)   
        model = SVC(**algo["hyper_param"])
    else:
        raise ValueError("Algorithme pas reconnu")

    model.fit(X, y)
    return model, scaler 

def predictFromHisto(S, model,scaler=None):
    predictions = []
    for img in S:                                 
        x = np.array(img["X_histo"]).reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        y_pred = model.predict(x)[0]
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
    elif algo["name"] == "RandomForest":
        model = RandomForestClassifier(**algo["hyper_param"])
    elif algo["name"] == "SVM" :
        model = Pipeline([("scaler", StandardScaler()),   ("clf",    SVC(**algo["hyper_param"]))   ])
    else:
        raise ValueError("Algorithme non reconnu")

    accuracies = cross_val_score(model, X, y, cv=k)
    return 1 - np.mean(accuracies)

path1 = r"Init/Mer" #remplacez le chemin par votre propre chemin ou pushez directement les 2 fichiers et remplacez ces 2 paths par les paths adéquats 
path2 = r"Init/Ailleurs"
S = buildSampleFromPath(path1, path2)
print("Nombre d'images chargees :", len(S))
algo = {"name": "SVM", "hyper_param": {"kernel": "rbf", "C": 1, "gamma": "scale"}}
model,scaler = fitFromHisto(S,algo)
predictFromHisto(S,model,scaler) 
y_true = [img["y_true_class"] for img in S]
y_pred = [img["y_predicted_class"] for img in S]

cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
tp, fn, fp, tn = cm.ravel()
print("--------- Matrice de Confusion ---------")
print(f"Vrais Positifs (Mer bien classée) : {tp}")
print(f"Vrais Négatifs (Ailleurs bien classé) : {tn}")
print(f"Faux Positifs (Ailleurs classé comme Mer) : {fp}")
print(f"Faux Négatifs (Mer classée comme Ailleurs) : {fn}")


# output of the false negatives and positives to manually check
false_neg_dir = "False_Negatives"
false_pos_dir = "False_Positives"
os.makedirs(false_neg_dir, exist_ok=True)
os.makedirs(false_pos_dir, exist_ok=True)

# Mer classée comme Ailleurs
for img in S:
    if img["y_true_class"] == 1 and img["y_predicted_class"] == -1:
        dest = os.path.join(false_neg_dir, os.path.basename(img["name_path"]))
        shutil.copy(img["name_path"], dest)

# Ailleurs classé comme Mer
for img in S:
    if img["y_true_class"] == -1 and img["y_predicted_class"] == 1:
        dest = os.path.join(false_pos_dir, os.path.basename(img["name_path"]))
        shutil.copy(img["name_path"], dest)

ee = erreurempirique(S)
er = crossValidationError(S, algo, k=5)

print(f"Erreur empirique (EE) : {ee:.4f}")
print(f"Erreur réelle (ER) : {er:.4f}")