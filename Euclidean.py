import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
from metric_learn import LFDA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Putanje do instrumenata

data_path = "C:/Users/User/Desktop/diplomski rad/DATA2/Train_submission"
instruments = ["drum","guitar", "piano", "violin"]  # 4 instrumenta


# 2. Funkcija za izračunavanje MFCC + ZCR + Spectral Centroid + Spectral Bandwidth + Spectral Roll-off
no_k = 16
def extract_features(file_path, n_mfcc=no_k):
    y, sr = librosa.load(file_path, duration=3)

    # === MFCC ===
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # === Zero Crossing Rate ===
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # === Spectral Centroid ===
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = np.mean(sc)
    sc_std = np.std(sc)

    # === Spectral Bandwidth ===
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sb_mean = np.mean(sb)
    sb_std = np.std(sb)

    # === Spectral Roll-off ===
    sr_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    sr_mean = np.mean(sr_roll)
    sr_std = np.std(sr_roll)

    # === Kombinuj sve u jedan vektor ===
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,           # 13 + 13 = 26
        [zcr_mean, zcr_std],           # 2
        [sc_mean, sc_std],             # 2
        [sb_mean, sb_std],             # 2
        [sr_mean, sr_std]              # 2
    ])

    return feature_vector



# 3. Učitavanje svih fajlova

X = []  # feature matrica
y = []  # labele (naziv instrumenta)

for label, instr in enumerate(instruments):
    folder = os.path.join(data_path, instr)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)
print("Shape y:", y.shape)




from sklearn.preprocessing import StandardScaler

# === NORMALIZACIJA (standardizacija) ===
scaler = StandardScaler()
X = scaler.fit_transform(X)



# Mapa engleski → srpski
instrument_map = {
    "drum": "bubnjevi",
    "guitar": "gitara",
    "piano": "klavir",
    "violin": "violina"
}

# Kreiraj listu srpskih naziva prema redosledu u instruments
instruments_srpski = [instrument_map[i] for i in instruments]


'''

# === STATISTIKA PRE BALANSIRANJA===
counts = [np.sum(y == i) for i in range(len(instruments))]

# Bar chart sa brojevima na stubovima
plt.figure(figsize=(8, 5))
bars = plt.bar(instruments_srpski, counts, color=['red','blue','green','orange'])
plt.title("Broj primera po instrumentu")
plt.ylabel("Broj fajlova")

# Dodaj brojeve iznad stubova
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()
'''



from imblearn.over_sampling import SMOTE

# === Balansiranje klasa SMOTE-om ===
smote = SMOTE(sampling_strategy='all', random_state=42)  # random_state za reproduktivnost
#smote = SMOTE(sampling_strategy='all')
X, y = smote.fit_resample(X, y)

print("Nakon SMOTE-a, shape X:", X.shape)

# Broj primera po klasi sa imenima instrumenata
counts = [np.sum(y == i) for i in range(len(instruments))]
counts_dict = {instr: count for instr, count in zip(instruments_srpski, counts)}

print("Broj primera po klasi nakon SMOTE-a:",
      [f"{instr}:{count}" for instr, count in counts_dict.items()])

'''
# === STATISTIKA ===
counts = [np.sum(y == i) for i in range(len(instruments))]

# Bar chart sa brojevima na stubovima
plt.figure(figsize=(8, 5))
#bars = plt.bar(instruments, counts, color=['red','blue','green','orange'])
bars = plt.bar(instruments_srpski, counts, color=['red', 'blue', 'green', 'orange'])
plt.title("Broj primera po instrumentu")
plt.ylabel("Broj fajlova")

# Dodaj brojeve iznad stubova
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()

# Pie chart
plt.figure(figsize=(6,6))
#plt.pie(counts, labels=instruments, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
plt.pie(counts, labels=instruments_srpski, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
plt.title("Udeo primera po instrumentima")
plt.show()
'''
#------------------------------------------------------------------------------------------------


# 4. LDA redukcija 2D
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)


# 5. LFDA redukcija 2D
lfda = LFDA(n_components=2, k=5)
X_lfda = lfda.fit_transform(X, y)


# 6. Crtanje poređenja
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
colors = ['red', 'blue', 'green', 'orange']

# LDA 2D
for i, instr in enumerate(instruments_srpski):
    axs[0].scatter(X_lda[y==i, 0], X_lda[y==i, 1], label=instr, alpha=0.7, color=colors[i])
axs[0].set_title("LDA redukcija MFCC 2D")
axs[0].set_xlabel("LDA1")
axs[0].set_ylabel("LDA2")
axs[0].legend()

# LFDA 2D
for i, instr in enumerate(instruments_srpski):
    axs[1].scatter(X_lfda[y==i, 0], X_lfda[y==i, 1], label=instr, alpha=0.7, color=colors[i])
axs[1].set_title("LFDA redukcija MFCC 2D")
axs[1].set_xlabel("LFDA1")
axs[1].set_ylabel("LFDA2")
axs[1].legend()

plt.tight_layout()
plt.show()



# LDA redukcija 3D
lda3 = LinearDiscriminantAnalysis(n_components=3)
X_lda3 = lda3.fit_transform(X, y)

# LFDA redukcija 3D
lfda3 = LFDA(n_components=3, k=5)
X_lfda3 = lfda3.fit_transform(X, y)


# Crtanje 3D poređenja
fig = plt.figure(figsize=(16, 7))

# LDA 3D subplot
ax1 = fig.add_subplot(121, projection='3d')
for i, instr in enumerate(instruments_srpski):
    ax1.scatter(
        X_lda3[y==i, 0],
        X_lda3[y==i, 1],
        X_lda3[y==i, 2],
        label=instr,
        alpha=0.7,
        s=50,
        color=colors[i]
    )
ax1.set_title("LDA redukcija MFCC 3D")
ax1.set_xlabel("LDA1")
ax1.set_ylabel("LDA2")
ax1.set_zlabel("LDA3")
ax1.legend()

# LFDA 3D subplot
ax2 = fig.add_subplot(122, projection='3d')
for i, instr in enumerate(instruments_srpski):
    ax2.scatter(
        X_lfda3[y==i, 0],
        X_lfda3[y==i, 1],
        X_lfda3[y==i, 2],
        label=instr,
        alpha=0.7,
        s=50,
        color=colors[i]
    )
ax2.set_title("LFDA redukcija MFCC 3D")
ax2.set_xlabel("LFDA1")
ax2.set_ylabel("LFDA2")
ax2.set_zlabel("LFDA3")
ax2.legend()

plt.tight_layout()
plt.show()



#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------




# --- FILTRACIJA TEST FAJLOVA (više labela podržano) ---
import pandas as pd
import os
import numpy as np

# --- Učitavanje putanja ---
test_path = "C:/Users/User/Desktop/diplomski rad/DATA2/Test_submission/Test_submission"
metadata_path = "C:/Users/User/Desktop/diplomski rad/DATA2/Metadata_Test.csv"

# --- Učitavanje metapodataka iz CSV fajla ---
metadata = pd.read_csv(metadata_path)
print("Učitano redova iz metadata:", len(metadata))

# --- Kreiranje test skupa ---
X_test = []
y_test = []

# Lista instrumenata (isti redosled kao u treniranju)
#instruments = ["piano", "violin", "drum", "guitar"]
instruments = ["drum","guitar", "piano", "violin"]  # 4 instrumenta


for _, row in metadata.iterrows():
    file_name = row["FileName"]
    label = row["Class"].strip().lower()

    wav_path = os.path.join(test_path, file_name)
    if not os.path.exists(wav_path):
        print(f"Upozorenje: fajl {file_name} nije pronađen, preskačem.")
        continue

    # Ekstrakcija karakteristika (MFCC)
    features = extract_features(wav_path)

    X_test.append(features)
    y_test.append(instruments.index(label))

# Pretvaranje u numpy nizove
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Test shape:", X_test.shape)
print("Broj različitih labela:", len(np.unique(y_test)))

# === Normalizacija test skupa istim scalerom ===
X_test = scaler.transform(X_test)


# -----------------------------------------------------------------------------------------------------------------



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# KNN

# === KNN na originalnim MFCC ===

#knn_raw = KNeighborsClassifier(n_neighbors=3) # euklidska
knn_raw = KNeighborsClassifier(n_neighbors=3, metric='euclidean') # euklidska
#knn_raw = KNeighborsClassifier(n_neighbors=3, weights='distance') # euklidska oteženjena

knn_raw.fit(X, y)
y_pred_raw = knn_raw.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)
print("KNN tačnost (MFCC):", round(acc_raw, 3))
cm_raw = confusion_matrix(y_test, y_pred_raw)
disp = ConfusionMatrixDisplay(cm_raw, display_labels=instruments_srpski)
disp.plot(cmap="Blues")

disp.ax_.tick_params(axis='x', labelsize=12)
disp.ax_.tick_params(axis='y', labelsize=12)

# Povećaj brojeve unutar same matrice
for text in disp.text_.ravel():
    text.set_fontsize(13)          # promena veličine fonta
    text.set_fontweight('bold')    # po želji – podebljaj tekst

# Nazivi osa
disp.ax_.set_xlabel("Predviđena klasa", fontsize=13)
disp.ax_.set_ylabel("Stvarna klasa", fontsize=13)

plt.title("KNN - Originalni MFCC (Euklidska distanca)", fontsize=14, fontweight = 'bold')
#plt.title("KNN - Originalni MFCC (Euklidska distanca, oteženjena)", fontsize=13, fontweight = 'bold')
plt.show()



# === KNN na LDA redukovanim podacima ===
# Moramo transformisati i test skup pomoću istog lda modela
X_test_lda = lda.transform(X_test)
knn_lda = KNeighborsClassifier(n_neighbors=3)
#knn_lda = KNeighborsClassifier(n_neighbors=3, weights='distance')

knn_lda.fit(X_lda, y)
y_pred_lda = knn_lda.predict(X_test_lda)
acc_lda = accuracy_score(y_test, y_pred_lda)
print("KNN tačnost (LDA):", round(acc_lda, 3))
cm_lda = confusion_matrix(y_test, y_pred_lda)
disp = ConfusionMatrixDisplay(cm_lda, display_labels=instruments_srpski)
disp.plot(cmap="Oranges")

disp.ax_.tick_params(axis='x', labelsize=12)
disp.ax_.tick_params(axis='y', labelsize=12)

# Povećaj brojeve unutar same matrice
for text in disp.text_.ravel():
    text.set_fontsize(13)          # promena veličine fonta
    text.set_fontweight('bold')    # po želji – podebljaj tekst

# Nazivi osa
disp.ax_.set_xlabel("Predviđena klasa", fontsize=13)
disp.ax_.set_ylabel("Stvarna klasa", fontsize=13)

plt.title("KNN - LDA redukcija (Euklidska distanca)", fontsize=14, fontweight = 'bold')
#plt.title("KNN - LDA redukcija (Euklidska distanca, oteženjena)", fontsize=13, fontweight = 'bold')
plt.show()



# === KNN na LFDA redukovanim podacima ===
# Moramo transformisati i test skup pomoću istog lfda modela
X_test_lfda = lfda.transform(X_test)
knn_lfda = KNeighborsClassifier(n_neighbors=3)
#knn_lfda = KNeighborsClassifier(n_neighbors=3, weights='distance')

knn_lfda.fit(X_lfda, y)
y_pred_lfda = knn_lfda.predict(X_test_lfda)
acc_lfda = accuracy_score(y_test, y_pred_lfda)
print("KNN tačnost (LFDA):", round(acc_lfda, 3))
cm_lfda = confusion_matrix(y_test, y_pred_lfda)
disp = ConfusionMatrixDisplay(cm_lfda, display_labels=instruments_srpski)
disp.plot(cmap="Greens")

disp.ax_.tick_params(axis='x', labelsize=12)
disp.ax_.tick_params(axis='y', labelsize=12)

# Povećaj brojeve unutar same matrice
for text in disp.text_.ravel():
    text.set_fontsize(13)          # promena veličine fonta
    text.set_fontweight('bold')    # po želji – podebljaj tekst

# Nazivi osa
disp.ax_.set_xlabel("Predviđena klasa", fontsize=13)
disp.ax_.set_ylabel("Stvarna klasa", fontsize=13)

plt.title("KNN - LFDA redukcija (Euklidska distanca)", fontsize=14, fontweight = 'bold')
#plt.title("KNN - LFDA redukcija (Euklidska distanca, oteženjena)", fontsize=13, fontweight = 'bold')
plt.show()



# === Poređenje tačnosti ===
print("\n--- Rezime tačnosti (k=3) ---")
print(f"Originalni MFCC: {acc_raw:.3f}")
print(f"LDA redukcija:   {acc_lda:.3f}")
print(f"LFDA redukcija:  {acc_lfda:.3f}")

#---------------------------------------------------------------------------------------------------------------

# === TESTIRANJE RAZLIČITIH VREDNOSTI k ===
k_values = [1, 3, 5, 7, 9, 11]
results = []

print("\n--- Eksperiment: promena broja komšija (k) ---")

for k in k_values:
    # === Originalni MFCC ===
    #knn_raw = KNeighborsClassifier(n_neighbors=k)
    knn_raw = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn_raw.fit(X, y)
    acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))

    # === LDA ===
    X_test_lda = lda.transform(X_test)
    #knn_lda = KNeighborsClassifier(n_neighbors=k)
    knn_lda = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn_lda.fit(X_lda, y)
    acc_lda = accuracy_score(y_test, knn_lda.predict(X_test_lda))

    # === LFDA ===
    X_test_lfda = lfda.transform(X_test)
    #knn_lfda = KNeighborsClassifier(n_neighbors=k)
    knn_lfda = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn_lfda.fit(X_lfda, y)
    acc_lfda = accuracy_score(y_test, knn_lfda.predict(X_test_lfda))

    results.append((k, acc_raw, acc_lda, acc_lfda))
    print(f"k={k:2d} → MFCC={acc_raw:.3f}, LDA={acc_lda:.3f}, LFDA={acc_lfda:.3f}")

# === Crtanje rezultata ===
results = np.array(results)
plt.figure(figsize=(8,5))
plt.plot(results[:,0], results[:,1], '-o', label="MFCC", color='blue')
plt.plot(results[:,0], results[:,2], '-o', label="LDA", color='orange')
plt.plot(results[:,0], results[:,3], '-o', label="LFDA", color='green')
plt.xlabel("Broj komšija (k)", fontsize=13)
plt.ylabel("Tačnost", fontsize=13)
plt.title("Uticaj broja komšija na tačnost KNN klasifikatora", fontsize=14, fontweight = 'bold')
plt.legend()
plt.grid(True)
plt.show()
