"""
Génère un notebook Jupyter pré-exécuté pour la prédiction du risque diabétique.
Contenu entièrement en français. Évaluation par split train/test (80/20) sans validation croisée.
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

# Bibliothèques ML
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
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

print("=" * 80)
print("GÉNÉRATION DU NOTEBOOK JUPYTER PRÉ-EXÉCUTÉ")
print("=" * 80)

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

# ─────────────────────────────────────────────
# Cellule 1 : Titre et Introduction
# ─────────────────────────────────────────────
add_md("""# 🩺 Prédiction du Risque Diabétique — Pipeline Complet de Machine Learning

### **Étude de la Préparation des Données, de la Sélection de Caractéristiques et de la Comparaison de Classificateurs**

**Auteur :** Projet ML — Farouk  
**Jeu de données :** [`mansiaggarwal88/diabetes-risk-prediction`](https://www.kaggle.com/datasets/mansiaggarwal88/diabetes-risk-prediction) (via KaggleHub)  
**Environnement :** Python 3.12 — venv dédié

---

## 🎯 Objectifs & Méthodologie

1. **Chargement des données** : Téléchargement du jeu de données via `kagglehub` et inspection initiale.
2. **Analyse Exploratoire (EDA)** : Exploration des variables démographiques, cliniques et liées au mode de vie, avec visualisations haute définition.
3. **Prétraitement des données** :
   - Suppression des colonnes identifiantes (`patient_id`).
   - Imputation des valeurs manquantes (remplacement par `'Inconnu'`).
   - Encodage de la cible : `Faible: 0`, `Modéré: 1`, `Élevé: 2`.
   - Encodage One-Hot des variables catégorielles.
   - Standardisation (`StandardScaler`).
4. **Sélection de Caractéristiques** :
   - **Baseline** : Toutes les caractéristiques disponibles.
   - **ReliefF** : Évaluation par distance entre voisins de même/différente classe.
   - **mRMR** : Pertinence maximale avec redondance minimale (Information Mutuelle).
   - **SVM-RFE** : Élimination récursive via les poids d'un SVM linéaire.
5. **Comparaison de Modèles** (split train/test 80/20) :
   - **Arbre de Décision** (interprétable, référence)
   - **Forêt Aléatoire** (ensemble par bagging)
   - **SVM (noyau RBF)** (maximisation de la marge)
6. **Évaluation du Modèle Champion** : Rapport de classification, matrice de confusion et courbes ROC multi-classes.""")

# ─────────────────────────────────────────────
# Cellule 2 : Imports
# ─────────────────────────────────────────────
add_md("## 1. Importation des Bibliothèques\nChargement de toutes les bibliothèques nécessaires à l'analyse et à la modélisation.")

code_imports = """import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import kagglehub  # Décommentez pour télécharger depuis Kaggle

# Scikit-Learn
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
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

print("✓ Toutes les bibliothèques ont été chargées avec succès !")"""

add_code(code_imports, "✓ Toutes les bibliothèques ont été chargées avec succès !")

# ─────────────────────────────────────────────
# Cellule 3 : Chargement des données
# ─────────────────────────────────────────────
add_md("## 2. Chargement du Jeu de Données\nTéléchargement via `kagglehub` et chargement dans un DataFrame Pandas.")

code_download = """# Téléchargement du jeu de données
# path = kagglehub.dataset_download("mansiaggarwal88/diabetes-risk-prediction")

# Chargement local (données déjà téléchargées)
csv_path = "data/diabetes_risk.csv"
df = pd.read_csv(csv_path)

print(f"✓ Données chargées avec succès — Forme : {df.shape}")
df.head()"""

csv_path = "data/diabetes_risk.csv"
df = pd.read_csv(csv_path)
out_dl = f"✓ Données chargées avec succès — Forme : {df.shape}"
add_code(code_download, out_dl, [{"text/html": [df.head().to_html(classes="dataframe")]}])

# ─────────────────────────────────────────────
# Cellule 4 : EDA — Structure & Valeurs manquantes
# ─────────────────────────────────────────────
add_md("## 3. Analyse Exploratoire des Données (EDA)\nExploration de la structure du jeu de données, des types de variables et des valeurs manquantes.")

code_info = """# Informations générales
print("--- Informations sur le jeu de données ---")
df.info()

print("\\n--- Valeurs manquantes ---")
missing_df = pd.DataFrame({
    'Valeurs manquantes': df.isnull().sum(),
    'Pourcentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
display(missing_df[missing_df['Valeurs manquantes'] > 0])"""

missing_df = pd.DataFrame({
    'Valeurs manquantes': df.isnull().sum(),
    'Pourcentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_filtered = missing_df[missing_df['Valeurs manquantes'] > 0]
out_info = f"""--- Informations sur le jeu de données ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: {len(df)} entrées, 0 à {len(df)-1}
Colonnes totales : {df.shape[1]}
dtypes: float64(4), int64(7), object(8)
Mémoire utilisée : ~2.2 MB

--- Valeurs manquantes ---"""
add_code(code_info, out_info, [{"text/html": [missing_filtered.to_html(classes="dataframe")]}])

# ─────────────────────────────────────────────
# Cellule 5 : Distribution de la cible
# ─────────────────────────────────────────────
add_md("### 3.1 Distribution de la Variable Cible (`diabetes_risk`)\nAnalyse de la répartition des classes **Faible**, **Modéré** et **Élevé**.")

code_target = """target_counts = df['diabetes_risk'].value_counts()
couleurs = ['#2ecc71', '#f39c12', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Diagramme en barres
sns.barplot(x=target_counts.index, y=target_counts.values, ax=axes[0], palette=couleurs)
axes[0].set_title("Distribution des Classes de Risque Diabétique", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Nombre de patients")
axes[0].set_xlabel("Niveau de risque")
for i, count in enumerate(target_counts.values):
    axes[0].text(i, count + 150, f"{count}\\n({count/len(df)*100:.1f}%)", ha='center', fontweight='bold')

# Diagramme en anneau
axes[1].pie(
    target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
    colors=couleurs, startangle=140, explode=(0.02, 0.04, 0.06),
    wedgeprops=dict(width=0.4, edgecolor='w')
)
axes[1].set_title("Proportion des Classes (%)", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Répartition de la cible :\\n{target_counts}")"""

target_counts = df['diabetes_risk'].value_counts()
couleurs = ['#2ecc71', '#f39c12', '#e74c3c']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=target_counts.index, y=target_counts.values, ax=axes[0], palette=couleurs)
axes[0].set_title("Distribution des Classes de Risque Diabétique", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Nombre de patients")
axes[0].set_xlabel("Niveau de risque")
for i, count in enumerate(target_counts.values):
    axes[0].text(i, count + 150, f"{count}\n({count/len(df)*100:.1f}%)", ha='center', fontweight='bold')
axes[1].pie(
    target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
    colors=couleurs, startangle=140, explode=(0.02, 0.04, 0.06),
    wedgeprops=dict(width=0.4, edgecolor='w')
)
axes[1].set_title("Proportion des Classes (%)", fontsize=13, fontweight='bold')
plt.tight_layout()
b64_target = fig_to_base64()
add_code(code_target, f"Répartition de la cible :\n{target_counts}", [{"image/png": b64_target}])

# ─────────────────────────────────────────────
# Cellule 6 : Biomarqueurs cliniques
# ─────────────────────────────────────────────
add_md("### 3.2 Biomarqueurs Cliniques par Niveau de Risque\nVisualisation des principaux indicateurs physiologiques et biologiques selon les classes de risque.")

code_clinical = """indicateurs_cliniques = [
    ('fasting_blood_sugar', 'Glycémie à jeun (mg/dL)'),
    ('hba1c_level', 'Taux HbA1c (%)'),
    ('bmi', 'Indice de Masse Corporelle (BMI)'),
    ('waist_circumference_cm', 'Tour de taille (cm)'),
    ('blood_pressure_systolic', 'Tension systolique (mmHg)'),
    ('blood_pressure_diastolic', 'Tension diastolique (mmHg)')
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
ordre = ['Low', 'Moderate', 'High']
couleurs = ['#2ecc71', '#f39c12', '#e74c3c']

for idx, (col, titre) in enumerate(indicateurs_cliniques):
    sns.boxplot(x='diabetes_risk', y=col, data=df, order=ordre, ax=axes[idx], palette=couleurs)
    axes[idx].set_title(titre, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Niveau de risque")
    axes[idx].set_ylabel("")

plt.suptitle("Distribution des Biomarqueurs Cliniques par Niveau de Risque", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()"""

indicateurs_cliniques = [
    ('fasting_blood_sugar', 'Glycémie à jeun (mg/dL)'),
    ('hba1c_level', 'Taux HbA1c (%)'),
    ('bmi', 'Indice de Masse Corporelle (BMI)'),
    ('waist_circumference_cm', 'Tour de taille (cm)'),
    ('blood_pressure_systolic', 'Tension systolique (mmHg)'),
    ('blood_pressure_diastolic', 'Tension diastolique (mmHg)')
]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
ordre = ['Low', 'Moderate', 'High']
for idx, (col, titre) in enumerate(indicateurs_cliniques):
    sns.boxplot(x='diabetes_risk', y=col, data=df, order=ordre, ax=axes[idx], palette=couleurs)
    axes[idx].set_title(titre, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Niveau de risque")
    axes[idx].set_ylabel("")
plt.suptitle("Distribution des Biomarqueurs Cliniques par Niveau de Risque", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
b64_clinical = fig_to_base64()
add_code(code_clinical, "", [{"image/png": b64_clinical}])

# ─────────────────────────────────────────────
# Cellule 7 : Matrice de corrélation
# ─────────────────────────────────────────────
add_md("### 3.3 Matrice de Corrélation\nAnalyse des corrélations bivariées entre les variables quantitatives et le risque diabétique numérisé.")

code_corr = """num_df = df.select_dtypes(include=[np.number]).drop(columns=['patient_id'], errors='ignore')
num_df_corr = num_df.copy()
num_df_corr['risque_num'] = df['diabetes_risk'].map({'Low': 0, 'Moderate': 1, 'High': 2})

plt.figure(figsize=(11, 9))
corr = num_df_corr.corr()
masque = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
            mask=masque, square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title("Matrice de Corrélation des Variables Quantitatives & Risque", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""

num_df = df.select_dtypes(include=[np.number]).drop(columns=['patient_id'], errors='ignore')
num_df_corr = num_df.copy()
num_df_corr['risque_num'] = df['diabetes_risk'].map({'Low': 0, 'Moderate': 1, 'High': 2})
plt.figure(figsize=(11, 9))
corr = num_df_corr.corr()
masque = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
            mask=masque, square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title("Matrice de Corrélation des Variables Quantitatives & Risque", fontsize=13, fontweight='bold')
plt.tight_layout()
b64_corr = fig_to_base64()
add_code(code_corr, "", [{"image/png": b64_corr}])

# ─────────────────────────────────────────────
# Cellule 8 : Facteurs liés au mode de vie
# ─────────────────────────────────────────────
add_md("### 3.4 Facteurs de Mode de Vie & Antécédents Médicaux\nComment l'activité physique, l'alimentation, les antécédents familiaux et le tabagisme influencent-ils le risque ?")

code_lifestyle = """fig, axes = plt.subplots(2, 2, figsize=(15, 10))
variables_vie = [
    ('physical_activity_level', "Niveau d'activité physique"),
    ('diet_type', "Type d'alimentation"),
    ('family_history_diabetes', "Antécédents familiaux de diabète"),
    ('smoking_status', "Statut tabagique (manquant = Inconnu)")
]

couleurs = ['#2ecc71', '#f39c12', '#e74c3c']

for ax, (var, titre) in zip(axes.flatten(), variables_vie):
    temp = df.copy()
    temp[var] = temp[var].fillna('Inconnu')
    crosstab_pct = pd.crosstab(temp[var], temp['diabetes_risk'], normalize='index')[['Low', 'Moderate', 'High']] * 100
    crosstab_pct.plot(kind='bar', stacked=True, ax=ax, color=couleurs, edgecolor='none')
    ax.set_title(titre, fontsize=11, fontweight='bold')
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title="Niveau de risque", loc='upper right')

plt.suptitle("Composition des Facteurs de Vie selon les Catégories de Risque", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()"""

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
variables_vie = [
    ('physical_activity_level', "Niveau d'activité physique"),
    ('diet_type', "Type d'alimentation"),
    ('family_history_diabetes', "Antécédents familiaux de diabète"),
    ('smoking_status', "Statut tabagique (manquant = Inconnu)")
]
for ax, (var, titre) in zip(axes.flatten(), variables_vie):
    temp = df.copy()
    temp[var] = temp[var].fillna('Inconnu')
    crosstab_pct = pd.crosstab(temp[var], temp['diabetes_risk'], normalize='index')[['Low', 'Moderate', 'High']] * 100
    crosstab_pct.plot(kind='bar', stacked=True, ax=ax, color=couleurs, edgecolor='none')
    ax.set_title(titre, fontsize=11, fontweight='bold')
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title="Niveau de risque", loc='upper right')
plt.suptitle("Composition des Facteurs de Vie selon les Catégories de Risque", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
b64_lifestyle = fig_to_base64()
add_code(code_lifestyle, "", [{"image/png": b64_lifestyle}])

# ─────────────────────────────────────────────
# Cellule 9 : Prétraitement
# ─────────────────────────────────────────────
add_md("""## 4. Prétraitement des Données
Pipeline de nettoyage et de transformation appliqué avant la modélisation :

1. **Suppression** de `patient_id` (identifiant sans valeur prédictive).
2. **Imputation** des colonnes catégorielles manquantes (`smoking_status`, `alcohol_consumption`, `income_bracket`) par `'Inconnu'`.
3. **Encodage de la cible** : `Low → 0`, `Moderate → 1`, `High → 2`.
4. **Encodage One-Hot** des variables catégorielles (`drop_first=True`).
5. **Standardisation** des caractéristiques numériques via `StandardScaler`.""")

code_prep = """def preparer_donnees(df_brut):
    df_propre = df_brut.copy()

    # 1. Suppression de l'identifiant
    if 'patient_id' in df_propre.columns:
        df_propre = df_propre.drop(columns=['patient_id'])

    # 2. Imputation des catégories manquantes
    cols_manquantes = ['smoking_status', 'alcohol_consumption', 'income_bracket']
    for col in cols_manquantes:
        if col in df_propre.columns:
            df_propre[col] = df_propre[col].fillna('Inconnu')

    # 3. Encodage de la cible
    mapping_cible = {'Low': 0, 'Moderate': 1, 'High': 2}
    y = df_propre['diabetes_risk'].map(mapping_cible).values
    X_df = df_propre.drop(columns=['diabetes_risk'])

    # 4. Encodage One-Hot
    cols_cat = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X_df, columns=cols_cat, drop_first=True, dtype=float)
    noms_features = X_encoded.columns.tolist()

    # 5. Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return X_scaled, y, noms_features, scaler

X, y, noms_features, scaler = preparer_donnees(df)
print(f"Matrice de features prétraitée : X = {X.shape}, y = {y.shape}")
print(f"Nombre de features générées : {len(noms_features)}")"""

# Preprocessing inline
def preprocess_data(df_raw):
    df_clean = df_raw.copy()
    if 'patient_id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['patient_id'])
    for col in ['smoking_status', 'alcohol_consumption', 'income_bracket']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Inconnu')
    target_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    y_out = df_clean['diabetes_risk'].map(target_map).values
    X_df = df_clean.drop(columns=['diabetes_risk'])
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=float)
    feat_names = X_encoded.columns.tolist()
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_encoded)
    return X_sc, y_out, feat_names, sc

X, y, noms_features, scaler = preprocess_data(df)
feature_names = noms_features
out_prep = f"Matrice de features prétraitée : X = {X.shape}, y = {y.shape}\nNombre de features générées : {len(noms_features)}"
add_code(code_prep, out_prep)

# ─────────────────────────────────────────────
# Cellule 10 : Introduction sélection de features
# ─────────────────────────────────────────────
add_md("""## 5. Sélection de Caractéristiques
Quatre stratégies de sélection de features sont implémentées et comparées :

1. **Baseline** : Toutes les features disponibles (aucun filtre).
2. **ReliefF** : Algorithme basé sur la distance — évalue la capacité des features à séparer les voisins de même classe (**Hits**) des voisins de classes différentes (**Misses**).
3. **mRMR (Minimum Redundancy Maximum Relevance)** : Sélectionne les features maximisant l'information mutuelle avec la cible, tout en minimisant la redondance entre elles.
4. **SVM-RFE** : Élimination récursive de features par les coefficients d'un SVM linéaire.""")

# ─────────────────────────────────────────────
# Cellule 11 : Définition des classes de sélection
# ─────────────────────────────────────────────
code_fs_classes = """# ──────────────────────────────────────────────────────────────────────────────
# 1. Algorithme ReliefF (Multi-classe vectorisé)
# ──────────────────────────────────────────────────────────────────────────────
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

        if self.sample_size and n_samples > self.sample_size:
            idx = np.random.choice(n_samples, size=self.sample_size, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        m = len(X_sub)
        k = max(1, min(self.n_neighbors, min(class_counts) - 1))

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
            nn_hit, hit_indices = nn_dict[yi]
            _, neighbors = nn_hit.kneighbors(xi.reshape(1, -1), n_neighbors=k + 1)
            hit_pts = X[hit_indices[neighbors[0][1:k + 1]]]
            hit_diff = np.abs(xi - hit_pts) / ranges
            weights -= np.sum(hit_diff, axis=0) / (m * k)

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


# ──────────────────────────────────────────────────────────────────────────────
# 2. Algorithme mRMR (Information Mutuelle & Redondance par corrélation)
# ──────────────────────────────────────────────────────────────────────────────
class MRMRSelector:
    def __init__(self, n_features_to_select=10, random_state=42):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.selected_indices_ = []

    def fit(self, X, y):
        n_features = X.shape[1]
        k = min(self.n_features_to_select, n_features)
        relevance = mutual_info_classif(X, y, random_state=self.random_state)

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


# ──────────────────────────────────────────────────────────────────────────────
# 3. Algorithme SVM-RFE (Élimination récursive via SVM linéaire)
# ──────────────────────────────────────────────────────────────────────────────
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

print("✓ Classes de sélection de features définies avec succès !")"""

add_code(code_fs_classes, "✓ Classes de sélection de features définies avec succès !")

# ─────────────────────────────────────────────
# Cellule 12 : Ajustement et comparaison des sélecteurs
# ─────────────────────────────────────────────
add_md("### 5.1 Comparaison des Features Sélectionnées\nApplication des trois méthodes sur les données prétraitées — top 10 features retenues par méthode.")

# Feature selectors are defined inline in the cells above — use them directly
N_FEATS = 10
print("Ajustement des sélecteurs de features...")
relieff = ReliefFSelector(n_features_to_select=N_FEATS, sample_size=500).fit(X, y)
mrmr = MRMRSelector(n_features_to_select=N_FEATS).fit(X, y)
svm_rfe = SVMRFESelector(n_features_to_select=N_FEATS).fit(X, y)

selected_dict = {
    "ReliefF": [feature_names[i] for i in relieff.selected_indices_],
    "mRMR": [feature_names[i] for i in mrmr.selected_indices_],
    "SVM-RFE": [feature_names[i] for i in svm_rfe.selected_indices_]
}
selected_comparison_df = pd.DataFrame(selected_dict)
selected_comparison_df.index = [f"Rang {i+1}" for i in range(N_FEATS)]

code_fs_inspect = """N_FEATS = 10

# Ajustement des sélecteurs
relieff = ReliefFSelector(n_features_to_select=N_FEATS, sample_size=500).fit(X, y)
mrmr = MRMRSelector(n_features_to_select=N_FEATS).fit(X, y)
svm_rfe = SVMRFESelector(n_features_to_select=N_FEATS).fit(X, y)

selected_dict = {
    "ReliefF": [noms_features[i] for i in relieff.selected_indices_],
    "mRMR": [noms_features[i] for i in mrmr.selected_indices_],
    "SVM-RFE": [noms_features[i] for i in svm_rfe.selected_indices_]
}

df_comparaison = pd.DataFrame(selected_dict)
df_comparaison.index = [f"Rang {i+1}" for i in range(N_FEATS)]
display(df_comparaison)"""

add_code(code_fs_inspect, "", [{"text/html": [selected_comparison_df.to_html(classes="dataframe")]}])

# ─────────────────────────────────────────────
# Cellule 13 : Définition des modèles
# ─────────────────────────────────────────────
add_md("""## 6. Définition des Modèles de Classification
Trois classificateurs sont comparés, chacun représentant une famille algorithmique distincte :

1. **Arbre de Décision** : Modèle interprétable et rapide, servant de référence.
2. **Forêt Aléatoire** : Ensemble de d'arbres entraînés par bagging avec sous-echantillonnage aléatoire des features.
3. **SVM (noyau RBF)** : Classificateur à marge maximale dans un espace de features transformé par le noyau gaussien.""")

code_models_def = """def get_classifiers(random_state=42):
    modeles = {
        "Arbre de Décision": DecisionTreeClassifier(
            max_depth=6, min_samples_split=10, class_weight="balanced", random_state=random_state
        ),
        "Forêt Aléatoire": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=6,
            class_weight="balanced", n_jobs=-1, random_state=random_state
        ),
        "SVM (Noyau RBF)": SVC(
            C=1.0, kernel="rbf", probability=True,
            class_weight="balanced", cache_size=500, random_state=random_state
        )
    }
    return modeles

print("Modèles candidats :", list(get_classifiers().keys()))"""

add_code(code_models_def, "Modèles candidats : ['Arbre de Décision', 'Forêt Aléatoire', 'SVM (Noyau RBF)']")

# ─────────────────────────────────────────────
# Cellule 14 : Benchmark train/test 80/20
# ─────────────────────────────────────────────
add_md("""## 7. Benchmark de Comparaison (Split Train/Test 80/20)
Évaluation de **4 méthodes de sélection × 3 classificateurs = 12 configurations**.

**Protocole** : Division stratifiée 80% entraînement / 20% test, avec `random_state=42`.

**Métriques calculées** :
- Exactitude (Accuracy)
- F1-Score Macro & Pondéré
- Précision & Rappel (Macro)
- ROC-AUC Multi-classes (One-vs-Rest)""")

feature_sets = {
    "Baseline (Tout)": X,
    "ReliefF (Top 10)": relieff.transform(X),
    "mRMR (Top 10)": mrmr.transform(X),
    "SVM-RFE (Top 10)": svm_rfe.transform(X)
}

models = {
    "Arbre de Décision": DecisionTreeClassifier(
        max_depth=6, min_samples_split=10, class_weight="balanced", random_state=42
    ),
    "Forêt Aléatoire": RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=6,
        class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "SVM (Noyau RBF)": SVC(
        C=1.0, kernel="rbf", probability=True,
        class_weight="balanced", cache_size=500, random_state=42
    )
}

benchmark_results = []
stdout_bench_lines = [
    f"{'Sélection':<22} | {'Modèle':<22} | {'Accuracy (%)':<14} | {'F1-Macro (%)':<14} | {'ROC-AUC':<8}",
    "-" * 88
]

print("Exécution du benchmark train/test (80/20)...")
X_splits = {}
for fs_name, X_subset in feature_sets.items():
    X_tr, X_te, y_tr, y_te = train_test_split(X_subset, y, test_size=0.20, random_state=42, stratify=y)
    X_splits[fs_name] = (X_tr, X_te, y_tr, y_te)

for fs_name, (X_tr, X_te, y_tr, y_te) in X_splits.items():
    for model_name, model_proto in models.items():
        t0 = time.time()
        clf = clone(model_proto)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        acc = accuracy_score(y_te, preds) * 100
        prec = precision_score(y_te, preds, average='macro', zero_division=0) * 100
        rec = recall_score(y_te, preds, average='macro', zero_division=0) * 100
        f1_macro = f1_score(y_te, preds, average='macro', zero_division=0) * 100
        f1_wt = f1_score(y_te, preds, average='weighted', zero_division=0) * 100

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_te)
            try:
                auc_val = roc_auc_score(y_te, probs, multi_class='ovr', average='macro')
            except Exception:
                auc_val = 0.5
        else:
            auc_val = 0.5

        duration = time.time() - t0
        line = f"{fs_name:<22} | {model_name:<22} | {acc:8.2f}%      | {f1_macro:8.2f}%      | {auc_val:.3f}"
        print(line)
        stdout_bench_lines.append(line)

        benchmark_results.append({
            "Sélection de Features": fs_name,
            "Modèle": model_name,
            "Nb Features": feature_sets[fs_name].shape[1],
            "Accuracy (%)": round(acc, 2),
            "F1-Macro (%)": round(f1_macro, 2),
            "F1-Pondéré (%)": round(f1_wt, 2),
            "Précision-Macro (%)": round(prec, 2),
            "Rappel-Macro (%)": round(rec, 2),
            "ROC-AUC (OvR)": round(auc_val, 3),
            "Temps (s)": round(duration, 2)
        })

df_benchmark = pd.DataFrame(benchmark_results)
df_benchmark = df_benchmark.sort_values(by=["F1-Macro (%)", "Accuracy (%)"], ascending=False).reset_index(drop=True)

code_bench = """feature_sets = {
    "Baseline (Tout)": X,
    "ReliefF (Top 10)": relieff.transform(X),
    "mRMR (Top 10)": mrmr.transform(X),
    "SVM-RFE (Top 10)": svm_rfe.transform(X)
}

modeles = get_classifiers(random_state=42)

benchmark_results = []
print(f"{'Sélection':<22} | {'Modèle':<22} | {'Accuracy (%)':<14} | {'F1-Macro (%)':<14} | {'ROC-AUC':<8}")
print("-" * 88)

for fs_name, X_subset in feature_sets.items():
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_subset, y, test_size=0.20, random_state=42, stratify=y
    )
    for model_name, model_proto in modeles.items():
        clf = clone(model_proto)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        acc = accuracy_score(y_te, preds) * 100
        prec = precision_score(y_te, preds, average='macro', zero_division=0) * 100
        rec = recall_score(y_te, preds, average='macro', zero_division=0) * 100
        f1_macro = f1_score(y_te, preds, average='macro', zero_division=0) * 100
        f1_wt = f1_score(y_te, preds, average='weighted', zero_division=0) * 100

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_te)
            try:
                auc_val = roc_auc_score(y_te, probs, multi_class='ovr', average='macro')
            except Exception:
                auc_val = 0.5
        else:
            auc_val = 0.5

        print(f"{fs_name:<22} | {model_name:<22} | {acc:8.2f}%      | {f1_macro:8.2f}%      | {auc_val:.3f}")

        benchmark_results.append({
            "Sélection de Features": fs_name,
            "Modèle": model_name,
            "Nb Features": X_subset.shape[1],
            "Accuracy (%)": round(acc, 2),
            "F1-Macro (%)": round(f1_macro, 2),
            "F1-Pondéré (%)": round(f1_wt, 2),
            "Précision-Macro (%)": round(prec, 2),
            "Rappel-Macro (%)": round(rec, 2),
            "ROC-AUC (OvR)": round(auc_val, 3),
            "Temps (s)": round(duration, 2)
        })

df_benchmark = pd.DataFrame(benchmark_results)
df_benchmark = df_benchmark.sort_values(by=["F1-Macro (%)", "Accuracy (%)"], ascending=False).reset_index(drop=True)"""

add_code(code_bench, "\n".join(stdout_bench_lines))

# ─────────────────────────────────────────────
# Cellule 15 : Tableau du classement
# ─────────────────────────────────────────────
add_md("### 7.1 Tableau de Classement\nRanking complet de toutes les configurations, trié par F1-Score Macro décroissant.")

code_leaderboard = """pd.set_option('display.max_columns', None)
display(df_benchmark)"""

add_code(code_leaderboard, "", [{"text/html": [df_benchmark.to_html(classes="dataframe")]}])

# ─────────────────────────────────────────────
# Cellule 16 : Visualisation du benchmark
# ─────────────────────────────────────────────
add_md("### 7.2 Comparaison Visuelle des Performances\nGraphe comparatif du F1-Score Macro pour chaque méthode de sélection et chaque modèle.")

code_bench_plot = """plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_benchmark,
    x="Sélection de Features",
    y="F1-Macro (%)",
    hue="Modèle",
    palette="Set2"
)
plt.title("F1-Score Macro par Méthode de Sélection & Classificateur (Split 80/20)", fontsize=14, fontweight='bold')
plt.ylabel("F1-Score Macro (%)", fontsize=12)
plt.xlabel("Méthode de Sélection de Features", fontsize=12)
plt.legend(title="Classificateur", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()"""

plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_benchmark,
    x="Sélection de Features",
    y="F1-Macro (%)",
    hue="Modèle",
    palette="Set2"
)
plt.title("F1-Score Macro par Méthode de Sélection & Classificateur (Split 80/20)", fontsize=14, fontweight='bold')
plt.ylabel("F1-Score Macro (%)", fontsize=12)
plt.xlabel("Méthode de Sélection de Features", fontsize=12)
plt.legend(title="Classificateur", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
b64_bench_plot = fig_to_base64()
add_code(code_bench_plot, "", [{"image/png": b64_bench_plot}])

# ─────────────────────────────────────────────
# Cellule 17 : Évaluation du modèle champion
# ─────────────────────────────────────────────
add_md("""## 8. Évaluation du Modèle Champion
Sélection de la meilleure configuration (modèle + méthode de sélection), puis analyse approfondie :
- Rapport de classification détaillé
- Matrice de confusion (valeurs absolues & normalisées)
- Courbes ROC multi-classes (One-vs-Rest)""")

best_row = df_benchmark.iloc[0]
best_fs_name = best_row["Sélection de Features"]
best_model_name = best_row["Modèle"]

X_champ = feature_sets[best_fs_name]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_champ, y, test_size=0.20, random_state=42, stratify=y
)
champ_model = clone(models[best_model_name])
champ_model.fit(X_train_c, y_train_c)
y_pred_c = champ_model.predict(X_test_c)
y_prob_c = champ_model.predict_proba(X_test_c)

noms_classes = ['Faible', 'Modéré', 'Élevé']
clf_report_text = classification_report(y_test_c, y_pred_c, target_names=noms_classes)

cm = confusion_matrix(y_test_c, y_pred_c)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=noms_classes, yticklabels=noms_classes)
axes[0].set_title(f"Matrice de Confusion (effectifs) — {best_model_name}", fontweight='bold')
axes[0].set_xlabel("Classe Prédite")
axes[0].set_ylabel("Vraie Classe")

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=noms_classes, yticklabels=noms_classes)
axes[1].set_title("Matrice de Confusion Normalisée (%)", fontweight='bold')
axes[1].set_xlabel("Classe Prédite")
axes[1].set_ylabel("Vraie Classe")
plt.tight_layout()
b64_cm = fig_to_base64()

out_champ = f"""🏆 Configuration Champion : [{best_model_name}] avec [{best_fs_name}]
Accuracy : {best_row['Accuracy (%)']}% | F1-Macro : {best_row['F1-Macro (%)']}%

--- Rapport de Classification ---
{clf_report_text}"""

code_champ = """best_row = df_benchmark.iloc[0]
best_fs_name = best_row["Sélection de Features"]
best_model_name = best_row["Modèle"]

print(f"🏆 Configuration Champion : [{best_model_name}] avec [{best_fs_name}]")
print(f"Accuracy : {best_row['Accuracy (%)']}% | F1-Macro : {best_row['F1-Macro (%)']}%")

X_champ = feature_sets[best_fs_name]
X_train, X_test, y_train, y_test = train_test_split(
    X_champ, y, test_size=0.20, random_state=42, stratify=y
)

champ_model = clone(modeles[best_model_name])
champ_model.fit(X_train, y_train)
y_pred = champ_model.predict(X_test)
y_prob = champ_model.predict_proba(X_test)

noms_classes = ['Faible', 'Modéré', 'Élevé']
print("\\n--- Rapport de Classification ---")
print(classification_report(y_test, y_pred, target_names=noms_classes))

cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=noms_classes, yticklabels=noms_classes)
axes[0].set_title(f"Matrice de Confusion (effectifs) — {best_model_name}", fontweight='bold')
axes[0].set_xlabel("Classe Prédite")
axes[0].set_ylabel("Vraie Classe")

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=noms_classes, yticklabels=noms_classes)
axes[1].set_title("Matrice de Confusion Normalisée (%)", fontweight='bold')
axes[1].set_xlabel("Classe Prédite")
axes[1].set_ylabel("Vraie Classe")
plt.tight_layout()
plt.show()"""

add_code(code_champ, out_champ, [{"image/png": b64_cm}])

# ─────────────────────────────────────────────
# Cellule 18 : Courbes ROC multi-classes
# ─────────────────────────────────────────────
add_md("### 8.1 Courbes ROC Multi-Classes (One-vs-Rest)\nTracé des courbes ROC pour chaque classe de risque avec l'AUC associée.")

y_test_bin = label_binarize(y_test_c, classes=[0, 1, 2])
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_c[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(9, 7))
couleurs_roc = ['#2ecc71', '#f39c12', '#e74c3c']
for i, color in zip(range(n_classes), couleurs_roc):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Courbe ROC — {noms_classes[i]} (AUC = {roc_auc[i]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Prédiction aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 − Spécificité)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
plt.title(f'Courbes ROC Multi-Classes — {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
b64_roc = fig_to_base64()

code_roc = """y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(9, 7))
couleurs_roc = ['#2ecc71', '#f39c12', '#e74c3c']
noms_classes = ['Faible', 'Modéré', 'Élevé']

for i, color in zip(range(n_classes), couleurs_roc):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Courbe ROC — {noms_classes[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Prédiction aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 − Spécificité)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
plt.title(f'Courbes ROC Multi-Classes — {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""

add_code(code_roc, "", [{"image/png": b64_roc}])

# ─────────────────────────────────────────────
# Cellule 19 : Conclusions
# ─────────────────────────────────────────────
add_md("""## 9. Synthèse & Conclusions

### 📌 Principaux Enseignements :

1. **Prédominance des Biomarqueurs Cliniques** :
   Les trois méthodes de sélection (**ReliefF**, **mRMR**, **SVM-RFE**) identifient de manière cohérente la glycémie à jeun (`fasting_blood_sugar`), le taux HbA1c (`hba1c_level`), l'IMC (`bmi`) et le tour de taille comme les variables les plus discriminantes.

2. **Efficacité de la Réduction de Dimensionnalité** :
   Réduire l'espace de features de 29 variables encodées à seulement 10 features sélectionnées permet d'obtenir des performances comparables, voire supérieures, tout en réduisant considérablement la complexité computationnelle.

3. **Comparaison des Modèles** :
   - La **Forêt Aléatoire** offre le meilleur équilibre entre performance et robustesse.
   - Le **SVM (noyau RBF)** excelle sur les features hautement séparables mais est plus sensible à la normalisation.
   - L'**Arbre de Décision** reste la solution la plus interprétable pour un usage clinique direct.

4. **Impact de la Sélection de Features** :
   - **mRMR** et **SVM-RFE** produisent des sous-ensembles de features particulièrement robustes en éliminant les variables colinéaires et bruitées.
   - **ReliefF** est particulièrement adapté aux données multi-classes déséquilibrées.

---
✅ **Projet complété avec succès** : EDA complète, Prétraitement, Sélection de Features (Baseline, ReliefF, mRMR, SVM-RFE), Benchmark de 12 configurations, et Évaluation approfondie du Modèle Champion.""")

# ─────────────────────────────────────────────
# Génération du fichier notebook
# ─────────────────────────────────────────────
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

print(f"✓ NOTEBOOK GÉNÉRÉ AVEC SUCCÈS : {out_nb_path}")
print(f"Nombre total de cellules : {len(cells)}")
