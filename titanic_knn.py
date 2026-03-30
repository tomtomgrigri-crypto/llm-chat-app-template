#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet NSI : Prévoir les survivants du Titanic
Algorithme des k plus proches voisins (KNN) - Version Optimisée
Intègre : Pipelines, Normalisation, Validation Croisée, Matrice de Confusion et Interface Interactive
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def charger_et_preparer_donnees(fichier_csv):
    """
    Charge le fichier CSV et prépare les données :
    - Sélection des colonnes pertinentes
    - Gestion des valeurs manquantes (déléguée au Pipeline)
    - Encodage des données catégorielles
    """
    # Chargement des données
    df = pd.read_csv(fichier_csv)
    
    print("=== Étape 1 : Analyse des colonnes ===")
    print(f"Colonnes disponibles : {list(df.columns)}")
    print(f"Nombre de passagers : {len(df)}")
    
    # Sélection des colonnes pertinentes
    # On garde : Pclass, Sex, Age, SibSp, Parch, Fare
    colonnes_a_garder = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    df = df[colonnes_a_garder + ['Survived']]
    
    print(f"\n=== Étape 2 : Nettoyage des données ===")
    print(f"Valeurs manquantes avant traitement :\n{df.isnull().sum()}")
    
    # Encodage de la colonne Sex (male=0, female=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    print("\n=== Données après encodage (aperçu) ===")
    print(df.head())
    
    return df

def creer_pipeline_knn():
    """
    Crée un pipeline complet incluant :
    1. L'imputation des valeurs manquantes (moyenne)
    2. La normalisation des données (StandardScaler) - CRUCIAL pour KNN
    3. Le classifieur KNN
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Remplit les trous (ex: Age)
        ('scaler', StandardScaler()),                 # Met à l'échelle (0-1 ou moyenne 0)
        ('knn', KNeighborsClassifier())               # Le modèle KNN
    ])
    return pipeline

def trouver_meilleur_k(X_train, y_train):
    """
    Utilise la validation croisée pour trouver le meilleur nombre de voisins (k)
    et teste la pondération par distance.
    """
    print("\n=== Étape 3 : Optimisation des hyperparamètres (Grid Search) ===")
    
    # Création du pipeline de base
    pipeline = creer_pipeline_knn()
    
    # Grille de paramètres à tester
    param_grid = {
        'knn__n_neighbors': list(range(1, 21)),       # Test de k=1 à k=20
        'knn__weights': ['uniform', 'distance']       # Test avec et sans pondération
    }
    
    # Configuration de la recherche
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5,                # Validation croisée à 5 plis
        scoring='accuracy',
        n_jobs=-1            # Utilise tous les cœurs du processeur
    )
    
    print("Recherche en cours... (cela peut prendre quelques secondes)")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Meilleurs paramètres trouvés : {grid_search.best_params_}")
    print(f"✓ Meilleur score en validation croisée : {grid_search.best_score_:.2%}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def evaluer_modele_complet(modele, X_test, y_test):
    """
    Évalue le modèle avec précision, matrice de confusion et rapport détaillé.
    """
    print("\n=== Étape 4 : Évaluation finale sur l'ensemble de test ===")
    
    # Prédiction
    y_pred = modele.predict(X_test)
    
    # Précision globale
    precision = accuracy_score(y_test, y_pred)
    print(f"Précision globale : {precision:.2%}")
    
    # Matrice de confusion
    print("\n--- Matrice de Confusion ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Légende :")
    print(f"  [0][0] = Vrais Négatifs (Prédit Mort, Réel Mort) : {cm[0][0]}")
    print(f"  [0][1] = Faux Positifs (Prédit Vivant, Réel Mort) : {cm[0][1]}")
    print(f"  [1][0] = Faux Négatifs (Prédit Mort, Réel Vivant) : {cm[1][0]}")
    print(f"  [1][1] = Vrais Positifs (Prédit Vivant, Réel Vivant) : {cm[1][1]}")
    
    # Rapport détaillé
    print("\n--- Rapport de Classification ---")
    target_names = ['Non Survivant', 'Survivant']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return y_pred, precision, cm

def afficher_graphique_optimisation(cv_results, best_params):
    """
    Affiche un graphique de la précision en fonction de k.
    """
    try:
        # Filtrer les résultats pour 'uniform' et 'distance' si on veut être précis, 
        # mais ici on prend le max pour chaque k toutes pondérations confondues pour la lisibilité
        # Ou plus simple : on reconstruit une courbe moyenne ou on affiche les points testés.
        
        # Pour simplifier dans ce script sans dépendances graphiques complexes :
        # On va juste extraire les scores moyens pour chaque combinaison
        means = cv_results['mean_test_score']
        params = cv_results['params']
        
        # Séparation des deux stratégies de poids pour le graphique
        k_uniform = []
        scores_uniform = []
        k_distance = []
        scores_distance = []
        
        for i, p in enumerate(params):
            if p['knn__weights'] == 'uniform':
                k_uniform.append(p['knn__n_neighbors'])
                scores_uniform.append(means[i])
            else:
                k_distance.append(p['knn__n_neighbors'])
                scores_distance.append(means[i])
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_uniform, scores_uniform, 'o-', label='Poids Uniforme')
        plt.plot(k_distance, 's-', label='Ponds par Distance')
        
        # Marquer le meilleur point
        best_k = best_params['knn__n_neighbors']
        best_w = best_params['knn__weights']
        plt.axvline(x=best_k, color='r', linestyle='--', label=f'Meilleur k={best_k} ({best_w})')
        
        plt.xlabel('Nombre de voisins (k)')
        plt.ylabel('Précision (Validation Croisée)')
        plt.title('Optimisation du nombre de voisins k')
        plt.grid(True)
        plt.legend()
        plt.savefig('optimisation_knn.png')
        print("\n[INFO] Graphique d'optimisation sauvegardé sous 'optimisation_knn.png'")
        plt.close()
    except Exception as e:
        print(f"\n[INFO] Impossible de générer le graphique (environnement sans affichage ?) : {e}")

def predire_survie_passager(modele, pclass, sex, age, sibsp, parch, fare):
    """
    Prédit la survie d'un nouveau passager en utilisant le pipeline complet.
    Le pipeline gère automatiquement l'imputation et la normalisation.
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
    prediction = modele.predict(nouveau_passager)[0]
    probabilites = modele.predict_proba(nouveau_passager)[0]
    
    return prediction, probabilites

def interface_interactive(modele):
    """
    Lance une petite boucle interactive pour tester des passagers.
    """
    print("\n" + "="*60)
    print("INTERFACE DE PRÉDICTION INTERACTIVE")
    print("Entrez les informations d'un passager pour prédire sa survie.")
    print("Tapez 'quit' pour quitter.")
    print("="*60)
    
    while True:
        try:
            print("\nNouveau passager :")
            inp = input("  Classe (1, 2, 3) : ")
            if inp == 'quit': break
            pclass = int(inp)
            
            sex = input("  Sexe (male/female) : ")
            if sex == 'quit': break
            
            age = float(input("  Age : "))
            sibsp = int(input("  Nb frères/sœurs/conjoints : "))
            parch = int(input("  Nb parents/enfants : "))
            fare = float(input("  Prix du billet : "))
            
            pred, proba = predire_survie_passager(modele, pclass, sex, age, sibsp, parch, fare)
            
            statut = "SURVIVANT" if pred == 1 else "NON SURVIVANT"
            print(f"\n>>> RÉSULTAT : {statut}")
            print(f"    Probabilité de survie : {proba[1]:.1%}")
            
        except ValueError:
            print("Erreur de saisie, veuillez recommencer.")
        except Exception as e:
            print(f"Une erreur est survenue : {e}")

def main():
    """
    Fonction principale du programme
    """
    print("=" * 70)
    print("PROJET NSI : PRÉVOIR LES SURVIVANTS DU TITANIC (VERSION OPTIMISÉE)")
    print("Pipeline : Imputation -> Normalisation -> KNN avec Validation Croisée")
    print("=" * 70)
    
    # 1. Charger et préparer les données
    df = charger_et_preparer_donnees('titanic.csv')
    
    # 2. Séparer données et label
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # 3. Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\nDonnées d'entraînement : {X_train.shape}")
    print(f"Données de test : {X_test.shape}")
    
    # 4. Trouver le meilleur modèle (Hyperparamètres + Validation Croisée)
    meilleur_modele, meilleurs_params, cv_results = trouver_meilleur_k(X_train, y_train)
    
    # 5. Afficher le graphique d'optimisation
    afficher_graphique_optimisation(cv_results, meilleurs_params)
    
    # 6. Évaluer le modèle final sur l'ensemble de test (Données jamais vues)
    y_pred, precision_finale, matrice_confusion = evaluer_modele_complet(meilleur_modele, X_test, y_test)
    
    # 7. Validation du seuil
    print("\n=== Étape 5 : Validation du seuil de performance ===")
    if precision_finale >= 0.65:
        print(f"✓ Modèle validé ! Précision de {precision_finale:.2%} >= 65%")
    else:
        print(f"⚠ Attention : Précision de {precision_finale:.2%} < 65%")
    
    # 8. Exemples rapides
    print("\n=== Étape 6 : Démonstration rapide ===")
    exemples = [
        {"pclass": 3, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": 7.0},
        {"pclass": 1, "sex": "female", "age": 30, "sibsp": 0, "parch": 0, "fare": 50.0},
        {"pclass": 3, "sex": "male", "age": 8, "sibsp": 1, "parch": 1, "fare": 15.0}
    ]
    
    for i, ex in enumerate(exemples):
        pred, proba = predire_survie_passager(meilleur_modele, **ex)
        statut = "Survivant" if pred == 1 else "Non survivant"
        print(f"Exemple {i+1} ({ex['sex']}, {ex['age']} ans, Cl.{ex['pclass']}) -> {statut} ({proba[1]:.1%} chance)")

    # 9. Lancer l'interface interactive (Optionnel, décommenter pour activer)
    # interface_interactive(meilleur_modele)
    
    print("\n" + "=" * 70)
    print("FIN DU PROGRAMME")
    print("Pour lancer l'interface interactive, décommentez la ligne à la fin du main().")
    print("=" * 70)
    
    return meilleur_modele, precision_finale

if __name__ == "__main__":
    modele, precision = main()
