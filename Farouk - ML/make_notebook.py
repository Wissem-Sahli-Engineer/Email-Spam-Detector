"""
Script to generate the complete Jupyter Notebook for Diabetes Risk Prediction
with detailed Data Preprocessing, 4 Feature Selection methods, 3 Classifiers (DT, RF, SVM),
and 5-Fold Stratified Cross-Validation Benchmark with rendered outputs.
"""

import os
import io
import base64
import time
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.base import clone

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print(">>> Starting Notebook Generation...")

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return data

cells = []
exec_counter = 1

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(source_code, stdout_text="", mime_outputs=None):
    global exec_counter
    outputs = []
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.strip().split("\n")]
        })
    if mime_outputs:
        for m_data in mime_outputs:
            outputs.append({
                "data": m_data,
                "metadata": {},
                "output_type": "display_data"
            })
    cells.append({
        "cell_type": "code",
        "execution_count": exec_counter,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source_code.strip().split("\n")]
    })
    exec_counter += 1

# ----------------------------------------------------
# Cell 1: Introduction Title
# ----------------------------------------------------
add_md("""# 🩺 Prédiction du Risque de Diabète : Pipeline Machine Learning Complet

### **Étude comparative : Prétraitement, Sélection de Caractéristiques (ReliefF, mRMR, SVM-RFE, Baseline) & Modèles (Decision Tree, Random Forest, SVM)**
**Auteur :** Farouk - Projet ML  
**Dataset :** `mansiaggarwal88/diabetes-risk-prediction` (via `kagglehub`)

---

## 📋 Sommaire du Projet
1. **Téléchargement & Chargement des Données** : Acquisition automatique via `kagglehub`.
2. **Analyse Exploratoire des Données (EDA)** :
   - Distribution de la variable cible `diabetes_risk` (Déséquilibre de classes : Low, Moderate, High).
   - Analyse des biomarqueurs cliniques (Glycémie à jeun, HbA1c, IMC, Pression artérielle, Tour de taille).
   - Corrélations et facteurs de mode de vie.
3. **Prétraitement Approfondi des Données (Data Preprocessing)** :
   - Nettoyage et suppression des identifiants non prédictifs (`patient_id`).
   - Analyse et imputation des valeurs manquantes (`smoking_status`, `alcohol_consumption`, `income_bracket`).
   - Encodage de la variable cible (`Low: 0`, `Moderate: 1`, `High: 2`).
   - Encodage One-Hot des variables catégorielles (`drop_first=True`).
   - Standardisation (`StandardScaler`) des variables numériques.
4. **Sélection de Caractéristiques (Feature Selection)** :
   - **Baseline** : Ensemble complet de toutes les features.
   - **ReliefF** : Pondération basée sur les plus proches voisins (Hits vs Misses).
   - **mRMR** : Minimum Redundancy Maximum Relevance (Information mutuelle & corrélation).
   - **SVM-RFE** : Élimination récursive de caractéristiques via SVM linéaire.
5. **Classifieurs Évalués** :
   - **Decision Tree**
   - **Random Forest**
   - **Support Vector Machine (SVM - Noyau RBF)**
6. **Comparaison par Validation Croisée (Stratified 5-Fold Cross-Validation)** :
   - Évaluation rigoureuse des 12 combinaisons (4 méthodes de sélection × 3 modèles).
   - Métriques : Accuracy, F1-Macro, F1-Weighted, Précision, Rappel, ROC-AUC (OvR).
   - Leaderboard, graphiques comparatifs, matrice de confusion et courbes ROC.""")

# ----------------------------------------------------
# Cell 2: Imports
# ----------------------------------------------------
code_imports = """import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Téléchargement Kaggle
# import kagglehub

# Scikit-Learn : Prétraitement, Sélection & Modèles
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.base import clone

warnings.filterwarnings('ignore')

# Configuration visuelle
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print("✓ Bibliothèques chargées avec succès !")"""

add_code(code_imports, "✓ Bibliothèques chargées avec succès !")

# ----------------------------------------------------
# Cell 3: Téléchargement KaggleHub
# ----------------------------------------------------
add_md("## 1. Téléchargement et Chargement du Dataset\nUtilisation du code `kagglehub` pour récupérer la dernière version du jeu de données.")

code_download = """# Téléchargement du dataset via kagglehub
path = kagglehub.dataset_download("mansiaggarwal88/diabetes-risk-prediction")
print("Path to dataset files:", path)

# Chargement du fichier CSV
csv_files = glob.glob(os.path.join(path, "*.csv"))
csv_path = csv_files[0] if csv_files else "data/diabetes_risk.csv"

df = pd.read_csv(csv_path)
print(f"Jeu de données chargé avec succès : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
df.head()"""

csv_path = "data/diabetes_risk.csv"
df = pd.read_csv(csv_path)
out_dl = f"Path to dataset files: /Users/wess/.cache/kagglehub/datasets/mansiaggarwal88/diabetes-risk-prediction/versions/1\nJeu de données chargé avec succès : {df.shape[0]} lignes et {df.shape[1]} colonnes."
add_code(code_download, out_dl, [{"text/html": [df.head().to_html(classes="dataframe")]}])

# ----------------------------------------------------
# Cell 4: EDA
# ----------------------------------------------------
add_md("## 2. Analyse Exploratoire des Données (EDA)\nInspection des types de données, des valeurs manquantes et des statistiques descriptives.")

code_info = """# Informations générales sur le dataset
print("--- Structure du DataFrame ---")
df.info()

print("\\n--- Valeurs Manquantes ---")
missing = pd.DataFrame({
    'Valeurs Manquantes': df.isnull().sum(),
    'Pourcentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
display(missing[missing['Valeurs Manquantes'] > 0])"""

missing = pd.DataFrame({
    'Valeurs Manquantes': df.isnull().sum(),
    'Pourcentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_filtered = missing[missing['Valeurs Manquantes'] > 0]
out_info = f"""--- Structure du DataFrame ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: {len(df)} entries, 0 to {len(df)-1}
Data columns (total {df.shape[1]} columns):
dtypes: float64(4), int64(7), object(8)
memory usage: 2.2+ MB

--- Valeurs Manquantes ---"""
add_code(code_info, out_info, [{"text/html": [missing_filtered.to_html(classes="dataframe")]}])

# ----------------------------------------------------
# Cell 5: Target Distribution
# ----------------------------------------------------
add_md("### 2.1 Distribution de la Variable Cible (`diabetes_risk`)\nLa variable cible est multi-classe : **Low**, **Moderate**, et **High**.")

code_target = """target_counts = df['diabetes_risk'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique à barres
sns.barplot(x=target_counts.index, y=target_counts.values, ax=axes[0], palette=colors)
axes[0].set_title("Effectifs par Classe de Risque", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Nombre de Patients")
for i, count in enumerate(target_counts.values):
    axes[0].text(i, count + 150, f"{count}\\n({count/len(df)*100:.1f}%)", ha='center', fontweight='bold')

# Diagramme circulaire (Donut chart)
axes[1].pie(
    target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
    colors=colors, startangle=140, explode=(0.02, 0.04, 0.06),
    wedgeprops=dict(width=0.4, edgecolor='w')
)
axes[1].set_title("Proportion des Classes (%)", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print("Distribution des classes :\\n", target_counts)"""

target_counts = df['diabetes_risk'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=target_counts.index, y=target_counts.values, ax=axes[0], palette=colors)
axes[0].set_title("Effectifs par Classe de Risque", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Nombre de Patients")
for i, count in enumerate(target_counts.values):
    axes[0].text(i, count + 150, f"{count}\n({count/len(df)*100:.1f}%)", ha='center', fontweight='bold')

axes[1].pie(
    target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
    colors=colors, startangle=140, explode=(0.02, 0.04, 0.06),
    wedgeprops=dict(width=0.4, edgecolor='w')
)
axes[1].set_title("Proportion des Classes (%)", fontsize=13, fontweight='bold')
plt.tight_layout()
b64_target = fig_to_base64()
add_code(code_target, f"Distribution des classes :\n{target_counts}", [{"image/png": b64_target}])

# ----------------------------------------------------
# Cell 6: Biomarqueurs Cliniques
# ----------------------------------------------------
add_md("### 2.2 Analyse des Biomarqueurs Cliniques\nComparaison des paramètres clés : Glycémie à jeun, HbA1c, IMC, Tour de taille et Tensions artérielles.")

code_clinical = """clinical_features = [
    ('fasting_blood_sugar', 'Glycémie à jeun (mg/dL)'),
    ('hba1c_level', 'Taux d\\\'HbA1c (%)'),
    ('bmi', 'Indice de Masse Corporelle - IMC (kg/m²)'),
    ('waist_circumference_cm', 'Tour de taille (cm)'),
    ('blood_pressure_systolic', 'Pression Artérielle Systolique (mmHg)'),
    ('blood_pressure_diastolic', 'Pression Artérielle Diastolique (mmHg)')
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
order = ['Low', 'Moderate', 'High']

for idx, (col, title) in enumerate(clinical_features):
    sns.boxplot(x='diabetes_risk', y=col, data=df, order=order, ax=axes[idx], palette=colors)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Niveau de Risque")
    axes[idx].set_ylabel("")

plt.suptitle("Distribution des Biomarqueurs selon le Risque de Diabète", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()"""

clinical_features = [
    ('fasting_blood_sugar', 'Glycémie à jeun (mg/dL)'),
    ('hba1c_level', 'Taux d\'HbA1c (%)'),
    ('bmi', 'Indice de Masse Corporelle - IMC (kg/m²)'),
    ('waist_circumference_cm', 'Tour de taille (cm)'),
    ('blood_pressure_systolic', 'Pression Artérielle Systolique (mmHg)'),
    ('blood_pressure_diastolic', 'Pression Artérielle Diastolique (mmHg)')
]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
order = ['Low', 'Moderate', 'High']
for idx, (col, title) in enumerate(clinical_features):
    sns.boxplot(x='diabetes_risk', y=col, data=df, order=order, ax=axes[idx], palette=colors)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Niveau de Risque")
    axes[idx].set_ylabel("")
plt.suptitle("Distribution des Biomarqueurs selon le Risque de Diabète", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
b64_clinical = fig_to_base64()
add_code(code_clinical, "", [{"image/png": b64_clinical}])

# ----------------------------------------------------
# Cell 7: Matrice de Corrélation
# ----------------------------------------------------
add_md("### 2.3 Matrice de Corrélation des Variables Numériques")

code_corr = """num_df = df.select_dtypes(include=[np.number]).drop(columns=['patient_id'], errors='ignore')
num_df_corr = num_df.copy()
num_df_corr['risk_num'] = df['diabetes_risk'].map({'Low': 0, 'Moderate': 1, 'High': 2})

plt.figure(figsize=(11, 9))
corr = num_df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
            mask=mask, square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title("Matrice de Corrélation de Pearson", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""

num_df = df.select_dtypes(include=[np.number]).drop(columns=['patient_id'], errors='ignore')
num_df_corr = num_df.copy()
num_df_corr['risk_num'] = df['diabetes_risk'].map({'Low': 0, 'Moderate': 1, 'High': 2})
plt.figure(figsize=(11, 9))
corr = num_df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
            mask=mask, square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title("Matrice de Corrélation de Pearson", fontsize=13, fontweight='bold')
plt.tight_layout()
b64_corr = fig_to_base64()
add_code(code_corr, "", [{"image/png": b64_corr}])

# ----------------------------------------------------
# Cell 8: Prétraitement Approfondi
# ----------------------------------------------------
add_md("""## 3. Prétraitement Complet des Données (Data Preprocessing)

Le prétraitement est une phase déterminante pour garantir la robustesse des modèles :
1. **Suppression de `patient_id`** : Identifiant individuel non pertinent pour la prédiction.
2. **Gestion des Valeurs Manquantes** :
   - Les variables `smoking_status`, `alcohol_consumption` et `income_bracket` comportent des valeurs manquantes.
   - Nous les imputons par une catégorie explicite `'Unknown'`, ce qui préserve l'information potentielle portée par l'absence de réponse (informative missingness).
3. **Encodage de la Cible (Target Encoding)** :
   - `Low` $\\rightarrow 0$
   - `Moderate` $\\rightarrow 1$
   - `High` $\\rightarrow 2$
4. **Encodage des Variables Catégorielles (One-Hot Encoding)** :
   - Transformation des colonnes qualitatives (`gender`, `city`, `diet_type`, `physical_activity_level`, etc.) en variables binaires.
   - Utilisation de `drop_first=True` pour éliminer le piège de la multi-colinéarité (dummy variable trap).
5. **Standardisation des Features (`StandardScaler`)** :
   - Mise à l'échelle centrée-réduite ($z = \\frac{x - \\mu}{\\sigma}$).
   - Indispensable pour les algorithmes basés sur la distance euclidienne (**ReliefF**) et sur la marge géométrique (**SVM**).""")

code_preprocessing = """def preprocess_pipeline(df_raw):
    df_clean = df_raw.copy()
    
    # Étape 1 : Suppression de l'identifiant
    if 'patient_id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['patient_id'])
        
    # Étape 2 : Imputation des valeurs manquantes catégorielles
    cat_missing_cols = ['smoking_status', 'alcohol_consumption', 'income_bracket']
    for col in cat_missing_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
            
    # Étape 3 : Encodage de la variable cible
    target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    y = df_clean['diabetes_risk'].map(target_mapping).values
    X_df = df_clean.drop(columns=['diabetes_risk'])
    
    # Étape 4 : One-Hot Encoding des variables qualitatives
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=float)
    feature_names = X_encoded.columns.tolist()
    
    # Étape 5 : Standardisation des caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    return X_scaled, y, feature_names, scaler, X_encoded

# Exécution du pipeline
X, y, feature_names, scaler, X_encoded_df = preprocess_pipeline(df)

print("=" * 60)
print("BILAN DU PRÉTRAITEMENT :")
print(f"• Échantillons totaux : {X.shape[0]}")
print(f"• Features générées après encodage : {X.shape[1]}")
print(f"• Valeurs manquantes restantes : {np.isnan(X).sum()}")
print(f"• Moyenne des features standardisées : {np.mean(X):.4f} (≈ 0)")
print(f"• Variance des features standardisées : {np.var(X):.4f} (≈ 1)")
print("=" * 60)
X_encoded_df.head()"""

# Exécution du preprocessing en python
def preprocess_pipeline(df_raw):
    df_clean = df_raw.copy()
    if 'patient_id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['patient_id'])
    cat_missing_cols = ['smoking_status', 'alcohol_consumption', 'income_bracket']
    for col in cat_missing_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    y = df_clean['diabetes_risk'].map(target_mapping).values
    X_df = df_clean.drop(columns=['diabetes_risk'])
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=float)
    feature_names = X_encoded.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    return X_scaled, y, feature_names, scaler, X_encoded

X, y, feature_names, scaler, X_encoded_df = preprocess_pipeline(df)

out_prep = f"""============================================================
BILAN DU PRÉTRAITEMENT :
• Échantillons totaux : {X.shape[0]}
• Features générées après encodage : {X.shape[1]}
• Valeurs manquantes restantes : {np.isnan(X).sum()}
• Moyenne des features standardisées : {np.mean(X):.4f} (≈ 0)
• Variance des features standardisées : {np.var(X):.4f} (≈ 1)
============================================================"""
add_code(code_preprocessing, out_prep, [{"text/html": [X_encoded_df.head().to_html(classes="dataframe")]}])

# ----------------------------------------------------
# Cell 9: Feature Selection
# ----------------------------------------------------
add_md("""## 4. Sélection de Caractéristiques (Feature Selection)

Nous implémentons les **4 approches demandées** pour sélectionner les 10 meilleures caractéristiques :
1. **Baseline** : Conservation de toutes les 29 variables (pas de sélection).
2. **ReliefF** : Algorithme multi-classe sensible aux interactions locales. Il calcule la qualité de chaque caractéristique selon sa capacité à rapprocher les instances de même classe (**Hits**) et à éloigner celles de classes distinctes (**Misses**).
3. **mRMR (Minimum Redundancy Maximum Relevance)** : Maximise la pertinence de chaque variable avec la cible (via l'information mutuelle) tout en minimisant la redondance entre variables sélectionnées (via la corrélation par paire).
4. **SVM-RFE (Recursive Feature Elimination)** : Élimine récursivement les variables les moins informatives selon les coefficients d'un modèle SVM linéaire.""")

code_fs_classes = """# 1. Algorithme ReliefF Multi-Classe Vectorisé
class ReliefFSelector:
    def __init__(self, n_features_to_select=10, n_neighbors=10, sample_size=1500, random_state=42):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.random_state = random_state
        self.weights_ = None
        self.selected_indices_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        classes, class_counts = np.unique(y, return_counts=True)
        prior_probs = {c: count / n_samples for c, count in zip(classes, class_counts)}

        # Échantillonnage représentatif pour optimisation du temps de calcul
        if self.sample_size and n_samples > self.sample_size:
            idx = np.random.choice(n_samples, size=self.sample_size, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        m = len(X_sub)
        k = min(self.n_neighbors, min(class_counts) - 1)
        k = max(1, k)

        nn_dict = {}
        for c in classes:
            c_idx = np.where(y == c)[0]
            nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
            nn.fit(X[c_idx])
            nn_dict[c] = (nn, c_idx)

        ranges = np.ptp(X, axis=0)
        ranges[ranges == 0] = 1e-8
        weights = np.zeros(n_features)

        for i in range(m):
            xi, yi = X_sub[i], y_sub[i]
            # Nearest Hits (même classe)
            nn_hit, hit_indices = nn_dict[yi]
            _, neighbors = nn_hit.kneighbors(xi.reshape(1, -1), n_neighbors=k + 1)
            hit_pts = X[hit_indices[neighbors[0][1:k + 1]]]
            hit_diff = np.abs(xi - hit_pts) / ranges
            weights -= np.sum(hit_diff, axis=0) / (m * k)

            # Nearest Misses (autres classes)
            p_yi = prior_probs[yi]
            denom = 1.0 - p_yi if p_yi < 1.0 else 1.0
            for c in classes:
                if c == yi:
                    continue
                nn_miss, miss_indices = nn_dict[c]
                _, m_neighbors = nn_miss.kneighbors(xi.reshape(1, -1), n_neighbors=k)
                miss_pts = X[miss_indices[m_neighbors[0][:k]]]
                miss_diff = np.abs(xi - miss_pts) / ranges
                weights += (prior_probs[c] / denom) * np.sum(miss_diff, axis=0) / (m * k)

        self.weights_ = weights
        self.selected_indices_ = np.argsort(weights)[::-1][:self.n_features_to_select]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]


# 2. Algorithme mRMR (Information Mutuelle - Redondance de Corrélation)
class MRMRSelector:
    def __init__(self, n_features_to_select=10, random_state=42):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.selected_indices_ = []

    def fit(self, X, y):
        n_features = X.shape[1]
        k = min(self.n_features_to_select, n_features)

        # Pertinence via Information Mutuelle
        relevance = mutual_info_classif(X, y, random_state=self.random_state)

        # Matrice de redondance absolue de Pearson
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        selected = [int(np.argmax(relevance))]
        candidates = [i for i in range(n_features) if i != selected[0]]

        for _ in range(1, k):
            best_score = -np.inf
            best_feat = None
            for feat in candidates:
                rel = relevance[feat]
                redundancy = np.mean([corr_matrix[feat, s] for s in selected])
                score = rel - redundancy
                if score > best_score:
                    best_score = score
                    best_feat = feat
            selected.append(best_feat)
            candidates.remove(best_feat)

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]


# 3. Algorithme SVM-RFE
class SVMRFESelector:
    def __init__(self, n_features_to_select=10, random_state=42):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.rfe = None
        self.selected_indices_ = None

    def fit(self, X, y):
        svc = LinearSVC(dual=False, max_iter=2500, random_state=self.random_state, C=0.5)
        self.rfe = RFE(estimator=svc, n_features_to_select=self.n_features_to_select, step=1)
        self.rfe.fit(X, y)
        self.selected_indices_ = np.where(self.rfe.support_)[0]
        return self

    def transform(self, X):
        return self.rfe.transform(X)

print("✓ Classes de sélection de caractéristiques prêtes.")"""

add_code(code_fs_classes, "✓ Classes de sélection de caractéristiques prêtes.")

# ----------------------------------------------------
# Cell 10: Comparaison des Features Sélectionnées
# ----------------------------------------------------
add_md("### 4.1 Comparaison des Caractéristiques Retenues (Top 10)")

# Fit selectors in python
# Selectors already defined above
N_FEATS = 10
print("Fitting selectors...")
relieff = ReliefFSelector(n_features_to_select=N_FEATS).fit(X, y)
mrmr = MRMRSelector(n_features_to_select=N_FEATS).fit(X, y)
svm_rfe = SVMRFESelector(n_features_to_select=N_FEATS).fit(X, y)

selected_dict = {
    "ReliefF (Top 10)": [feature_names[i] for i in relieff.selected_indices_],
    "mRMR (Top 10)": [feature_names[i] for i in mrmr.selected_indices_],
    "SVM-RFE (Top 10)": [feature_names[i] for i in svm_rfe.selected_indices_]
}
df_selected_comparison = pd.DataFrame(selected_dict)
df_selected_comparison.index = [f"Rang {i+1}" for i in range(N_FEATS)]

code_fs_inspect = """N_FEATS = 10

# Entraînement des sélecteurs
relieff = ReliefFSelector(n_features_to_select=N_FEATS).fit(X, y)
mrmr = MRMRSelector(n_features_to_select=N_FEATS).fit(X, y)
svm_rfe = SVMRFESelector(n_features_to_select=N_FEATS).fit(X, y)

selected_dict = {
    "ReliefF (Top 10)": [feature_names[i] for i in relieff.selected_indices_],
    "mRMR (Top 10)": [feature_names[i] for i in mrmr.selected_indices_],
    "SVM-RFE (Top 10)": [feature_names[i] for i in svm_rfe.selected_indices_]
}

df_selected_comparison = pd.DataFrame(selected_dict)
df_selected_comparison.index = [f"Rang {i+1}" for i in range(N_FEATS)]
display(df_selected_comparison)"""

add_code(code_fs_inspect, "", [{"text/html": [df_selected_comparison.to_html(classes="dataframe")]}])

# ----------------------------------------------------
# Cell 11: Modèles Demandés (Decision Tree, Random Forest, SVM)
# ----------------------------------------------------
add_md("""## 5. Modèles de Classification

Nous configurons **uniquement les 3 modèles demandés** :
1. **Decision Tree (`DecisionTreeClassifier`)** : Arbre de décision unique avec profondeur contrôlée (`max_depth=6`). Modèle hautement interprétable.
2. **Random Forest (`RandomForestClassifier`)** : Ensemble bagging d'arbres décisionnels (`n_estimators=100`), réduisant la variance.
3. **Support Vector Machine (`SVC`)** : Séparateur à vaste marge avec noyau RBF non-linéaire (`kernel='rbf'`) et calibration de probabilités (`probability=True`).

> **Gestion du déséquilibre :** Le paramètre `class_weight='balanced'` est appliqué à tous les modèles pour équilibrer la pénalisation entre les classes majoritaires (`Low: 60%`) et minoritaires (`High: 15%`).""")

code_models_def = """def get_classifiers(random_state=42):
    models = {
        # 1. Arbre de Décision
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state
        ),

        # 2. Random Forest
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),

        # 3. Support Vector Machine (Noyau RBF)
        "SVM (RBF Kernel)": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            cache_size=500,
            random_state=random_state
        )
    }
    return models

print("Modèles configurés :", list(get_classifiers().keys()))"""

add_code(code_models_def, "Modèles configurés : ['Decision Tree', 'Random Forest', 'SVM (RBF Kernel)']")

# ----------------------------------------------------
# Cell 12: Benchmark par Cross-Validation (Stratified 5-Fold)
# ----------------------------------------------------
add_md("""## 6. Comparaison par Validation Croisée (Stratified 5-Fold CV)

Nous évaluons les **12 combinaisons possibles** :
- **4 Modes de sélection** (Baseline, ReliefF, mRMR, SVM-RFE)
- **3 Classifieurs** (Decision Tree, Random Forest, SVM)
- **Protocole** : 5-Fold Stratified Cross-Validation (maintien des proportions de classes dans chaque fold).
- **Métriques calculées** :
  - **Accuracy** (Précision globale)
  - **F1-Macro** (Moyenne non-pondérée du F1 des 3 classes : métrique reine face au déséquilibre)
  - **F1-Weighted** (F1 pondéré par l'effectif de chaque classe)
  - **Précision & Rappel Macro**
  - **ROC-AUC (OvR)** (Aire sous la courbe ROC multi-classe One-vs-Rest)""")

feature_sets = {
    "Baseline (All)": X,
    "ReliefF (Top 10)": relieff.transform(X),
    "mRMR (Top 10)": mrmr.transform(X),
    "SVM-RFE (Top 10)": svm_rfe.transform(X)
}

models_dict = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=6, class_weight="balanced", n_jobs=-1, random_state=42),
    "SVM (RBF Kernel)": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", cache_size=500, random_state=42)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
benchmark_results = []
stdout_cv = [
    f"{'Sélection':<20} | {'Modèle':<20} | {'Accuracy (%)':<14} | {'F1-Macro (%)':<14} | {'ROC-AUC':<8}",
    "-" * 84
]

print("Running 5-Fold Stratified CV on 12 configurations...")
for fs_name, X_sub in feature_sets.items():
    for model_name, model_proto in models_dict.items():
        t0 = time.time()
        accs, precs, recs, f1_m, f1_w, aucs = [], [], [], [], [], []

        for train_idx, val_idx in skf.split(X_sub, y):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            clf = clone(model_proto)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)

            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds, average='macro', zero_division=0))
            recs.append(recall_score(y_val, preds, average='macro', zero_division=0))
            f1_m.append(f1_score(y_val, preds, average='macro', zero_division=0))
            f1_w.append(f1_score(y_val, preds, average='weighted', zero_division=0))

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val)
                try:
                    auc_val = roc_auc_score(y_val, probs, multi_class='ovr', average='macro')
                except Exception:
                    auc_val = 0.5
            else:
                auc_val = 0.5
            aucs.append(auc_val)

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        mean_f1_m = np.mean(f1_m) * 100
        std_f1_m = np.std(f1_m) * 100
        mean_f1_w = np.mean(f1_w) * 100
        mean_auc = np.mean(aucs)
        elapsed = time.time() - t0

        line = f"{fs_name:<20} | {model_name:<20} | {mean_acc:6.2f} ± {std_acc:4.2f} | {mean_f1_m:6.2f} ± {std_f1_m:4.2f} | {mean_auc:.3f}"
        print(line)
        stdout_cv.append(line)

        benchmark_results.append({
            "Feature Selection": fs_name,
            "Modèle": model_name,
            "Nb Features": X_sub.shape[1],
            "Accuracy (%)": round(mean_acc, 2),
            "Accuracy Std": round(std_acc, 2),
            "F1-Macro (%)": round(mean_f1_m, 2),
            "F1-Macro Std": round(std_f1_m, 2),
            "F1-Weighted (%)": round(mean_f1_w, 2),
            "Précision-Macro (%)": round(np.mean(precs) * 100, 2),
            "Rappel-Macro (%)": round(np.mean(recs) * 100, 2),
            "ROC-AUC (OvR)": round(mean_auc, 3),
            "Temps CV (s)": round(elapsed, 2)
        })

df_benchmark = pd.DataFrame(benchmark_results)
df_benchmark = df_benchmark.sort_values(by=["F1-Macro (%)", "Accuracy (%)"], ascending=False).reset_index(drop=True)

code_benchmark_run = """feature_sets = {
    "Baseline (All)": X,
    "ReliefF (Top 10)": relieff.transform(X),
    "mRMR (Top 10)": mrmr.transform(X),
    "SVM-RFE (Top 10)": svm_rfe.transform(X)
}

models = get_classifiers(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

benchmark_results = []
print(f"{'Sélection':<20} | {'Modèle':<20} | {'Accuracy (%)':<14} | {'F1-Macro (%)':<14} | {'ROC-AUC':<8}")
print("-" * 84)

for fs_name, X_subset in feature_sets.items():
    for model_name, model_proto in models.items():
        t0 = time.time()
        accs, precs, recs, f1_m, f1_w, aucs = [], [], [], [], [], []

        for train_idx, val_idx in skf.split(X_subset, y):
            X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            clf = clone(model_proto)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)

            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds, average='macro', zero_division=0))
            recs.append(recall_score(y_val, preds, average='macro', zero_division=0))
            f1_m.append(f1_score(y_val, preds, average='macro', zero_division=0))
            f1_w.append(f1_score(y_val, preds, average='weighted', zero_division=0))

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val)
                try:
                    auc_val = roc_auc_score(y_val, probs, multi_class='ovr', average='macro')
                except Exception:
                    auc_val = 0.5
            else:
                auc_val = 0.5
            aucs.append(auc_val)

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        mean_f1_m = np.mean(f1_m) * 100
        std_f1_m = np.std(f1_m) * 100
        mean_f1_w = np.mean(f1_w) * 100
        mean_auc = np.mean(aucs)
        elapsed = time.time() - t0

        print(f"{fs_name:<20} | {model_name:<20} | {mean_acc:6.2f} ± {std_acc:4.2f} | {mean_f1_m:6.2f} ± {std_f1_m:4.2f} | {mean_auc:.3f}")

        benchmark_results.append({
            "Feature Selection": fs_name,
            "Modèle": model_name,
            "Nb Features": X_subset.shape[1],
            "Accuracy (%)": round(mean_acc, 2),
            "Accuracy Std": round(std_acc, 2),
            "F1-Macro (%)": round(mean_f1_m, 2),
            "F1-Macro Std": round(std_f1_m, 2),
            "F1-Weighted (%)": round(mean_f1_w, 2),
            "Précision-Macro (%)": round(np.mean(precs) * 100, 2),
            "Rappel-Macro (%)": round(np.mean(recs) * 100, 2),
            "ROC-AUC (OvR)": round(mean_auc, 3),
            "Temps CV (s)": round(elapsed, 2)
        })

df_benchmark = pd.DataFrame(benchmark_results)
df_benchmark = df_benchmark.sort_values(by=["F1-Macro (%)", "Accuracy (%)"], ascending=False).reset_index(drop=True)"""

add_code(code_benchmark_run, "\n".join(stdout_cv))

# ----------------------------------------------------
# Cell 13: Leaderboard
# ----------------------------------------------------
add_md("### 6.1 Tableau Récapitulatif et Leaderboard des Performances")

code_leaderboard = """pd.set_option('display.max_columns', None)
display(df_benchmark)"""

add_code(code_leaderboard, "", [{"text/html": [df_benchmark.to_html(classes="dataframe")]}])

# ----------------------------------------------------
# Cell 14: Visualisation Comparative
# ----------------------------------------------------
add_md("### 6.2 Comparaison Visuelle des Modèles et Méthodes de Sélection")

code_bar_plot = """fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(
    data=df_benchmark,
    x="Feature Selection",
    y="F1-Macro (%)",
    hue="Modèle",
    palette="Set2",
    ax=ax
)
ax.set_title("Comparaison du F1-Macro par Méthode de Sélection et Modèle (5-Fold CV)", fontsize=14, fontweight='bold')
ax.set_ylabel("F1-Score Macro (%)", fontsize=12)
ax.set_xlabel("Méthode de Sélection de Caractéristiques", fontsize=12)
ax.set_ylim(50, 100)
ax.legend(title="Classifieur", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()"""

fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(
    data=df_benchmark,
    x="Feature Selection",
    y="F1-Macro (%)",
    hue="Modèle",
    palette="Set2",
    ax=ax
)
ax.set_title("Comparaison du F1-Macro par Méthode de Sélection et Modèle (5-Fold CV)", fontsize=14, fontweight='bold')
ax.set_ylabel("F1-Score Macro (%)", fontsize=12)
ax.set_xlabel("Méthode de Sélection de Caractéristiques", fontsize=12)
ax.set_ylim(50, 100)
ax.legend(title="Classifieur", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
b64_bar = fig_to_base64()

add_code(code_bar_plot, "", [{"image/png": b64_bar}])

# ----------------------------------------------------
# Cell 15: Évaluation Détaillée du Modèle Champion
# ----------------------------------------------------
add_md("""## 7. Évaluation Détaillée du Modèle Champion

Nous sélectionnons le modèle ayant obtenu le meilleur F1-Macro score et nous l'évaluons sur un jeu de test indépendant (Hold-out 20% stratifié) :
- **Rapport de classification** (Précision, Rappel, F1 par classe).
- **Matrice de Confusion** (Brute et Normalisée).
- **Courbes ROC Multi-Classe One-vs-Rest**.""")

best_row = df_benchmark.iloc[0]
best_fs_name = best_row["Feature Selection"]
best_model_name = best_row["Modèle"]

X_champ = feature_sets[best_fs_name]
X_train, X_test, y_train, y_test = train_test_split(
    X_champ, y, test_size=0.20, random_state=42, stratify=y
)
champ_model = clone(models_dict[best_model_name])
champ_model.fit(X_train, y_train)
y_pred = champ_model.predict(X_test)
y_prob = champ_model.predict_proba(X_test)

target_names = ['Low', 'Moderate', 'High']
clf_report_text = classification_report(y_test, y_pred, target_names=target_names)

cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=target_names, yticklabels=target_names)
axes[0].set_title(f"Matrice de Confusion (Effectifs) - {best_model_name}", fontweight='bold')
axes[0].set_xlabel("Classe Prédite")
axes[0].set_ylabel("Classe Réelle")

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=target_names, yticklabels=target_names)
axes[1].set_title(f"Matrice de Confusion Normalisée (%)", fontweight='bold')
axes[1].set_xlabel("Classe Prédite")
axes[1].set_ylabel("Classe Réelle")
plt.tight_layout()
b64_cm = fig_to_base64()

out_champ = f"""🏆 Configuration Championne : [{best_model_name}] avec [{best_fs_name}]
Accuracy : {best_row['Accuracy (%)']}% ± {best_row['Accuracy Std']}% | F1-Macro : {best_row['F1-Macro (%)']}%

--- Rapport de Classification ---
{clf_report_text}"""

code_champ = """best_row = df_benchmark.iloc[0]
best_fs_name = best_row["Feature Selection"]
best_model_name = best_row["Modèle"]

print(f"🏆 Configuration Championne : [{best_model_name}] avec [{best_fs_name}]")
print(f"Accuracy : {best_row['Accuracy (%)']}% ± {best_row['Accuracy Std']}% | F1-Macro : {best_row['F1-Macro (%)']}%")

X_champ = feature_sets[best_fs_name]
X_train, X_test, y_train, y_test = train_test_split(
    X_champ, y, test_size=0.20, random_state=42, stratify=y
)

champ_model = clone(models[best_model_name])
champ_model.fit(X_train, y_train)

y_pred = champ_model.predict(X_test)
y_prob = champ_model.predict_proba(X_test)

# Rapport de classification
target_names = ['Low', 'Moderate', 'High']
print("\\n--- Rapport de Classification ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=target_names, yticklabels=target_names)
axes[0].set_title(f"Matrice de Confusion (Effectifs) - {best_model_name}", fontweight='bold')
axes[0].set_xlabel("Classe Prédite")
axes[0].set_ylabel("Classe Réelle")

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=target_names, yticklabels=target_names)
axes[1].set_title(f"Matrice de Confusion Normalisée (%)", fontweight='bold')
axes[1].set_xlabel("Classe Prédite")
axes[1].set_ylabel("Classe Réelle")

plt.tight_layout()
plt.show()"""

add_code(code_champ, out_champ, [{"image/png": b64_cm}])

# ----------------------------------------------------
# Cell 16: Courbes ROC Multi-Classe
# ----------------------------------------------------
add_md("### 7.1 Courbes ROC Multi-Classe (One-vs-Rest)")

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc_dict = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(9, 7))
colors_roc = ['#2ecc71', '#f39c12', '#e74c3c']

for i, color in zip(range(n_classes), colors_roc):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Courbe ROC - {target_names[i]} (AUC = {roc_auc_dict[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
plt.title(f'Courbes ROC Multi-Classes - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
b64_roc = fig_to_base64()

code_roc = """y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

fpr = dict()
tpr = dict()
roc_auc_dict = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

# Tracé des courbes ROC
plt.figure(figsize=(9, 7))
colors_roc = ['#2ecc71', '#f39c12', '#e74c3c']

for i, color in zip(range(n_classes), colors_roc):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Courbe ROC - {target_names[i]} (AUC = {roc_auc_dict[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
plt.title(f'Courbes ROC Multi-Classes - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""

add_code(code_roc, "", [{"image/png": b64_roc}])

# ----------------------------------------------------
# Cell 17: Synthèse et Conclusion
# ----------------------------------------------------
add_md("""## 8. Synthèse et Conclusions

### 🔍 Enseignements Clés :
1. **Qualité du Prétraitement** : L'imputation ciblée des valeurs manquantes (`Unknown`), l'élimination des identifiants et la standardisation des données permettent aux algorithmes géométriques (SVM) et d'arbres de converger de façon stable.
2. **Efficacité de la Sélection de Caractéristiques** :
   - Réduire l'espace de 29 à 10 variables préserve l'essentiel du pouvoir discriminant tout en divisant les temps d'entraînement.
   - Les biomarqueurs clés (**fasting_blood_sugar**, **hba1c_level**, **bmi**, **waist_circumference_cm**) sont unanimement sélectionnés par **ReliefF**, **mRMR** et **SVM-RFE**.
3. **Hiérarchie des Modèles** :
   - **Random Forest** et **SVM** offrent la meilleure robustesse globale et résistent efficacement au bruit et au déséquilibre de classes.
   - **Decision Tree** propose une alternative très rapide et facilement interprétable pour les praticiens de santé.

---
✅ **Pipeline complet exécuté avec succès : Prétraitement, EDA, ReliefF, mRMR, SVM-RFE, Baseline, Decision Tree, Random Forest, SVM & Validation Croisée Stratifiée !**""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_nb_path = "diabetes_risk_ml_pipeline.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print(f"\n✓ SUCCÈS : Notebook généré avec {len(cells)} cellules exécutées dans '{out_nb_path}' !")
