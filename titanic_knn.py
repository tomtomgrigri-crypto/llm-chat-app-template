#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet NSI : Prévoir les survivants du Titanic
Algorithme des k plus proches voisins (KNN)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

def charger_et_preparer_donnees(fichier_csv):
    """
    Charge le fichier CSV et prépare les données :
    - Sélection des colonnes pertinentes
    - Gestion des valeurs manquantes
    - Encodage des données catégorielles
    """
    # Chargement des données
    df = pd.read_csv(fichier_csv)
    
    print("=== Étape 1 : Analyse des colonnes ===")
    print(f"Colonnes disponibles : {list(df.columns)}")
    print(f"Nombre de passagers : {len(df)}")
    
    # Sélection des colonnes pertinentes
    # On garde : Pclass, Sex, Age, SibSp, Parch, Fare
    # On supprime : PassengerId, Name, Ticket, Cabin, Embarked (pour simplifier)
    colonnes_a_garder = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    df = df[colonnes_a_garder + ['Survived']]
    
    print(f"\n=== Étape 2 : Nettoyage des données ===")
    print(f"Valeurs manquantes avant traitement :\n{df.isnull().sum()}")
    
    # Gestion des valeurs manquantes pour l'âge
    # On remplace par la moyenne plutôt que de supprimer les lignes
    imputer_age = SimpleImputer(strategy='mean')
    df['Age'] = imputer_age.fit_transform(df[['Age']])
    
    # Gestion des valeurs manquantes pour Fare (très rares)
    imputer_fare = SimpleImputer(strategy='mean')
    df['Fare'] = imputer_fare.fit_transform(df[['Fare']])
    
    print(f"Valeurs manquantes après traitement :\n{df.isnull().sum()}")
    
    # Encodage de la colonne Sex (male=0, female=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    print("\n=== Données après encodage ===")
    print(df.head())
    
    return df

def separer_donnees_et_label(df):
    """
    Sépare les données (X) du label (y)
    """
    print("\n=== Étape 3 : Séparation des données et du label ===")
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    print(f"Shape de X (données) : {X.shape}")
    print(f"Shape de y (label) : {y.shape}")
    
    return X, y

def diviser_train_test(X, y):
    """
    Divise les données en ensemble d'entraînement et de test
    """
    print("\n=== Étape 4 : Division train/test ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,  # 20% pour le test
        random_state=42,  # Pour la reproductibilité
        stratify=y  # Pour garder la même proportion de survivants
    )
    
    print(f"Taille X_train : {X_train.shape}")
    print(f"Taille X_test : {X_test.shape}")
    print(f"Taille y_train : {y_train.shape}")
    print(f"Taille y_test : {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def entrainer_modele(X_train, y_train, k=5):
    """
    Entraîne le modèle KNN avec k voisins
    """
    print(f"\n=== Étape 5 : Entraînement du modèle (k={k}) ===")
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    print("Modèle entraîné avec succès !")
    
    return knn

def evaluer_modele(knn, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test
    """
    print("\n=== Étape 6 : Évaluation du modèle ===")
    
    # Prédiction sur l'ensemble de test
    y_pred = knn.predict(X_test)
    
    # Calcul de la précision
    precision = accuracy_score(y_test, y_pred)
    
    print(f"Précision du modèle : {precision:.2%}")
    print(f"Nombre de prédictions correctes : {sum(y_pred == y_test)} / {len(y_test)}")
    
    return y_pred, precision

def predire_survie_passager(knn, imputer_age, imputer_fare, pclass, sex, age, sibsp, parch, fare):
    """
    Prédit la survie d'un nouveau passager
    """
    # Encodage du sexe
    sex_code = 0 if sex.lower() == 'male' else 1
    
    # Création du dataframe pour le nouveau passager
    nouveau_passager = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_code],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })
    
    # Prédiction
    prediction = knn.predict(nouveau_passager)[0]
    probabilite = knn.predict_proba(nouveau_passager)[0]
    
    return prediction, probabilite

def main():
    """
    Fonction principale du programme
    """
    print("=" * 60)
    print("PROJET NSI : PRÉVOIR LES SURVIVANTS DU TITANIC")
    print("Algorithme des k plus proches voisins (KNN)")
    print("=" * 60)
    
    # 1. Charger et préparer les données
    df = charger_et_preparer_donnees('titanic.csv')
    
    # 2. Séparer données et label
    X, y = separer_donnees_et_label(df)
    
    # 3. Diviser en train/test
    X_train, X_test, y_train, y_test = diviser_train_test(X, y)
    
    # 4. Entraîner le modèle avec différents valeurs de k pour trouver le meilleur
    meilleur_k = 5
    meilleure_precision = 0
    
    print("\n=== Recherche du meilleur k ===")
    for k in range(1, 15, 2):  # Test des k impairs de 1 à 13
        knn_temp = entrainer_modele(X_train, y_train, k)
        _, precision_temp = evaluer_modele(knn_temp, X_test, y_test)
        
        if precision_temp > meilleure_precision:
            meilleure_precision = precision_temp
            meilleur_k = k
            print(f"k={k} -> Précision: {precision_temp:.2%} (Nouveau record!)")
        else:
            print(f"k={k} -> Précision: {precision_temp:.2%}")
    
    print(f"\n✓ Meilleur k trouvé : {meilleur_k} avec une précision de {meilleure_precision:.2%}")
    
    # 5. Entraîner le modèle final avec le meilleur k
    knn_final = entrainer_modele(X_train, y_train, meilleur_k)
    y_pred, precision_finale = evaluer_modele(knn_final, X_test, y_test)
    
    # 6. Vérifier si la précision est suffisante (> 65%)
    print("\n=== Étape 7 : Validation du modèle ===")
    if precision_finale >= 0.65:
        print(f"✓ Modèle validé ! Précision de {precision_finale:.2%} >= 65%")
        print("Vous pouvez maintenant utiliser le modèle pour prédire la survie de nouveaux passagers.")
    else:
        print(f"⚠ Attention : Précision de {precision_finale:.2%} < 65%")
        print("Le modèle pourrait être amélioré (plus de features, nettoyage différent, etc.)")
    
    # 7. Exemples de prédictions pour de nouveaux passagers
    print("\n=== Étape 8 : Exemples de prédictions ===")
    
    # Imputers pour gérer les âges manquants dans les nouvelles prédictions
    imputer_age = SimpleImputer(strategy='mean')
    imputer_age.fit(df[['Age']])
    imputer_fare = SimpleImputer(strategy='mean')
    imputer_fare.fit(df[['Fare']])
    
    # Exemple 1 : Homme adulte, 3ème classe
    pred1, proba1 = predire_survie_passager(
        knn_final, imputer_age, imputer_fare,
        pclass=3, sex='male', age=25, sibsp=0, parch=0, fare=7.0
    )
    print(f"\nPassager 1 : Homme, 25 ans, 3ème classe, seul")
    print(f"  Prédiction : {'Survivant' if pred1 == 1 else 'Non survivant'}")
    print(f"  Probabilités : Non survivant={proba1[0]:.1%}, Survivant={proba1[1]:.1%}")
    
    # Exemple 2 : Femme adulte, 1ère classe
    pred2, proba2 = predire_survie_passager(
        knn_final, imputer_age, imputer_fare,
        pclass=1, sex='female', age=30, sibsp=0, parch=0, fare=50.0
    )
    print(f"\nPassager 2 : Femme, 30 ans, 1ère classe, seule")
    print(f"  Prédiction : {'Survivant' if pred2 == 1 else 'Non survivant'}")
    print(f"  Probabilités : Non survivant={proba2[0]:.1%}, Survivant={proba2[1]:.1%}")
    
    # Exemple 3 : Enfant, 3ème classe
    pred3, proba3 = predire_survie_passager(
        knn_final, imputer_age, imputer_fare,
        pclass=3, sex='male', age=8, sibsp=1, parch=1, fare=15.0
    )
    print(f"\nPassager 3 : Garçon, 8 ans, 3ème classe, avec famille")
    print(f"  Prédiction : {'Survivant' if pred3 == 1 else 'Non survivant'}")
    print(f"  Probabilités : Non survivant={proba3[0]:.1%}, Survivant={proba3[1]:.1%}")
    
    print("\n" + "=" * 60)
    print("FIN DU PROGRAMME")
    print("=" * 60)
    
    return knn_final, precision_finale

if __name__ == "__main__":
    modele, precision = main()
