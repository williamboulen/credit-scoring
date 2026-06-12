import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. CHARGEMENT DES DONNÉES
# Le fichier german.data utilise des espaces comme séparateurs et n'a pas de header.
# La 21ème colonne correspond à la cible (1 = Bon crédit, 2 = Mauvais crédit).

columns = [
    "status_checking",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

print("Chargement des données...")
df = pd.read_csv("german.data", sep=r"\s+", names=columns)

# Recodage de la cible : 1 (Bon) -> 0, et 2 (Mauvais) -> 1
# En credit scoring, on cherche généralement à prédire le défaut (1 = défaut)
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

# 2. SÉPARATION DES FEATURES ET DE LA CIBLE
X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

# Identification des types de colonnes
num_features = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]
cat_features = [col for col in X.columns if col not in num_features]

# Split Train / Test (80% / 20%) avec stratification pour gérer le déséquilibre
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. PRÉPROCESSING (PIPELINE TECHNIQUE)
# On normalise les variables numériques et on One-Hot encode les catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

print("Transformation des données...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 4. ENTRAÎNEMENT DU MODÈLE (RANDOM FOREST)
print("Entraînement du modèle de Credit Scoring...")
# Utilisation de class_weight='balanced' car le dataset compte 70% de bons et 30% de mauvais crédits
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
)
model.fit(X_train_processed, y_train)

# 5. ÉVALUATION DES PERFORMANCES
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

print("\n" + "=" * 50)
print("RÉSULTATS DU MODÈLE")
print("=" * 50)

print("\n🔹 Matrice de Confusion :")
print(confusion_matrix(y_test, y_pred))

print("\n🔹 Rapport de Classification :")
print(classification_report(y_test, y_pred, target_names=["Bon", "Mauvais"]))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"🔹 Score ROC AUC : {roc_auc:.4f}")
print("=" * 50)
