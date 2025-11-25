# ================================================================
#  Simpsons Classification - Xception + ViT + 20 Classifiers
#  Autor: Thiago & Gepeto 🟨🟦
# ================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# TensorFlow (Xception)
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input as preprocess_xception

# PyTorch ViT (timm)
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T


# ================================================================
#               CONFIGURAÇÕES IMPORTANTES
# ================================================================

TRAIN_DIR = "simpsons/Train"
TEST_DIR  = "simpsons/Valid"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Rodando no dispositivo: {DEVICE.upper()}")


# ================================================================
#               CARREGAR MODELOS DE DEEP FEATURES
# ================================================================

print("\nCarregando Xception...")
xception = Xception(weights="imagenet", include_top=False, pooling="avg")

print("Carregando ViT-B/16 (timm)...")
vit = timm.create_model("vit_base_patch16_224", pretrained=True)
vit.head = nn.Identity()
vit = vit.to(DEVICE)
vit.eval()

vit_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# ================================================================
#                   FUNÇÃO DE EXTRAÇÃO DE FEATURES
# ================================================================

def get_label_from_filename(fname):
    return re.match(r"[a-zA-Z]+", fname).group(0)

def extract_features(folder):
    X = []
    Y = []

    files = os.listdir(folder)
    print(f"\n🔍 Extraindo features de {folder} ({len(files)} arquivos)")

    for fname in tqdm(files):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        label = get_label_from_filename(fname)
        path  = os.path.join(folder, fname)

        img = cv2.imread(path)
        if img is None:
            print(f"❌ Erro ao abrir {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ----------- XCEPTION -----------
        img_x = cv2.resize(img, (299, 299))
        x_in = preprocess_xception(img_x.astype("float32"))
        feat_x = xception.predict(np.expand_dims(x_in, 0), verbose=0).flatten()

        # ----------- VIT -----------
        img_v = vit_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat_v = vit(img_v).cpu().numpy().flatten()

        # -------- CONCAT --------
        features = np.concatenate([feat_x, feat_v])
        X.append(features)
        Y.append(label)

    return np.array(X), np.array(Y)


# ================================================================
#                     CARREGAR O DATASET
# ================================================================

print("\n📥 Extraindo Features...")
X_train, y_train = extract_features(TRAIN_DIR)
X_test, y_test   = extract_features(TEST_DIR)

print("\n📊 Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ================================================================
#               ENCODER DE LABELS
# ================================================================

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)


# ================================================================
#               DEFINIÇÃO DOS 20 CLASSIFICADORES
# ================================================================

classifiers = {
    "SVM_rbf_C1":      SVC(kernel="rbf", C=1, probability=True),
    "SVM_rbf_C10":     SVC(kernel="rbf", C=10, probability=True),
    "SVM_poly":        SVC(kernel="poly", degree=3, probability=True),
    "SVM_sigmoid":     SVC(kernel="sigmoid", probability=True),
    "SVM_linear":      SVC(kernel="linear", probability=True),

    "KNN_3":  KNeighborsClassifier(n_neighbors=3),
    "KNN_5":  KNeighborsClassifier(n_neighbors=5),
    "KNN_7":  KNeighborsClassifier(n_neighbors=7),
    "KNN_9":  KNeighborsClassifier(n_neighbors=9),
    "KNN_11": KNeighborsClassifier(n_neighbors=11),

    "RF_50":        RandomForestClassifier(n_estimators=50),
    "RF_100":       RandomForestClassifier(n_estimators=100),
    "RF_200":       RandomForestClassifier(n_estimators=200),
    "RF_depth10":   RandomForestClassifier(max_depth=10),
    "RF_depth20":   RandomForestClassifier(max_depth=20),

    "MLP_128":      MLPClassifier(hidden_layer_sizes=(128,), max_iter=600),
    "MLP_256":      MLPClassifier(hidden_layer_sizes=(256,), max_iter=600),
    "MLP_512":      MLPClassifier(hidden_layer_sizes=(512,), max_iter=600),
    "MLP_2layers":  MLPClassifier(hidden_layer_sizes=(256,128), max_iter=600),
    "MLP_relu":     MLPClassifier(activation="relu", hidden_layer_sizes=(256,), max_iter=600),
}


# ================================================================
#               TREINAR OS 20 CLASSIFICADORES
# ================================================================

results = {}

print("\n🚀 Treinando 20 classificadores...\n")

for name, clf in classifiers.items():
    print(f"\n🔵 Treinando {name}...")
    clf.fit(X_train, y_train_enc)
    
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, pred)
    f1  = f1_score(y_test_enc, pred, average="weighted")

    results[name] = (acc * 100, f1 * 100)

    print(f"✔ {name}: ACC={acc*100:.2f}% | F1={f1*100:.2f}%")


# ================================================================
#                       ENSEMBLE FINAL
# ================================================================

print("\n🤝 Treinando Ensemble (voting soft)...")

ensemble = VotingClassifier(
    estimators=[(name, clf) for name, clf in classifiers.items()],
    voting="soft"
)

ensemble.fit(X_train, y_train_enc)
pred_ens = ensemble.predict(X_test)

acc_ens = accuracy_score(y_test_enc, pred_ens)
f1_ens  = f1_score(y_test_enc, pred_ens, average="weighted")

print(f"\n🌟 ENSEMBLE RESULTADOS:")
print(f"Accuracy: {acc_ens*100:.2f}%")
print(f"F1-score: {f1_ens*100:.2f}%")


# ================================================================
#                     MATRIZ DE CONFUSÃO
# ================================================================

cm = confusion_matrix(y_test_enc, pred_ens)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusão - Ensemble")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.colorbar()

# --- Colocar números nos quadrados ---
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="black", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("matriz_confusao_ensemble.png", dpi=300)
plt.show()

print("\n📁 Matriz de confusão salva em: matriz_confusao_ensemble.png")
