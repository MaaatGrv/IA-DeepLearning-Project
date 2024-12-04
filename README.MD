# Rapport : **GTZAN - Music Genre Classification**
Le programme pour la partie machine learning est dans Projet_IA_Deep_Learning_part/deep_learning/test_pierre.ipynb
## 1. Sujet choisi 
### Contexte  
Le dataset **GTZAN - Music Genre Classification** est une référence incontournable dans le domaine de la reconnaissance automatique des genres musicaux. Créé par George Tzanetakis en 2001, il contient 1000 extraits audio de 30 secondes répartis uniformément sur 10 genres.

### Problématique 
Cette étude vise à analyser les caractéristiques audio présentes dans ce dataset et à évaluer leur potentiel pour classer automatiquement les genres musicaux à l'aide de modèles d'apprentissage automatique.

### Objectif : 
 - Objectif: Développer un modèle prédictif performant, pour reconnaitre les genres des musiques. 
---


## 2. Données utilisées  
### Description des données  
[Donnez un aperçu des données utilisées : leur source, leur structure (nombre de lignes, colonnes), et leurs caractéristiques principales.]  
GTZAN Genre Classification Dataset
Ce jeu de données est composé de 1 000 extraits audio de 30 secondes chacun, répartis uniformément en 10 genres musicaux :

- **Blues**  
- **Classical**  
- **Country**  
- **Disco**  
- **Hip-hop**  
- **Jazz**  
- **Metal**  
- **Pop**  
- **Reggae**  
- **Rock**

Chaque genre contient 100 extraits audio.

### Préparation des données  
- Source des données : Dataset Kaggle : https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification   
- Nombre d’échantillons : 1000 extraits de 30 secondes  
- Caractéristiques principales :  voir csv : features_3_sec.csv  
- Normalisation/Standardisation : Principalement avec des means et vars

---

## 3. Méthodes de prétraitement pour le Machine Learning  
### Étapes principales   

- Étape 1 : Encodage des variables catégoriques.  
- Étape 2 : Division des données en ensembles d’entraînement, de validation et de test:   70/30

---

## 4. Architecture du modèle et processus d’entraînement pour le ML  
### Modèle utilisé  
Choix du Cross Gradient Booster après comparaison (accuracy de 90%)  

- **Type de modèle :** XGBoost  
- **Hyperparamètres principaux :**  
  - Learning rate = 0.05
  - N-estimator = 1000
- **Architecture :**  
  - A METTRE POUR LE DL (couches de neuronnes) 

### Processus d’entraînement pour le DL
- Taux d’apprentissage : 0.05    

---

## 5. Résultats 
- **Score de performance :** Précision: 0.9012345679012346  
  


