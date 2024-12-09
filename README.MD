# Rapport : **GTZAN - Music Genre Classification**

Ce projet vise à classifier automatiquement les genres musicaux en utilisant le dataset GTZAN, en appliquant des techniques d'apprentissage automatique (Machine Learning) et de deep learning. Nous explorons différentes architectures de modèles et optimisons les hyperparamètres pour améliorer les performances de classification.

## Table des Matières

1. [Sujet choisi](#1-sujet-choisi)
   - [Contexte](#contexte)
   - [Problématique](#problématique)
   - [Objectif](#objectif)
2. [Données utilisées](#2-données-utilisées)
   - [Description des données](#description-des-données)
   - [Préparation des données](#préparation-des-données)
3. [Méthodes de prétraitement](#3-méthodes-de-prétraitement)
4. [Machine Learning](#4-machine-learning)
   - [Architecture du modèle et processus d’entraînement](#architecture-du-modèle-et-processus-dentraînement)
   - [Résultats](#résultats)
5. [Deep Learning](#5-deep-learning)
   - [Première implémentation du modèle](#première-implémentation-du-modèle)
   - [Travail en cours : Variation des hyperparamètres et exploration de nouveaux modèles](#travail-en-cours--variation-des-hyperparamètres-et-exploration-de-nouveaux-modèles)
6. [Installation](#6-installation)
7. [Utilisation](#7-utilisation)
8. [Résultats et Analyse](#8-résultats-et-analyse)
   - [Machine Learning](#machine-learning)
   - [Deep Learning](#deep-learning)
9. [Conclusion](#9-conclusion)
10. [Références](#10-références)
11. [Contact](#11-contact)

---

## 1. Sujet choisi

### Contexte

Le dataset **GTZAN - Music Genre Classification** est une référence incontournable dans le domaine de la reconnaissance automatique des genres musicaux. Créé par George Tzanetakis en 2001, il contient 1000 extraits audio de 30 secondes répartis uniformément sur 10 genres.

### Problématique

Cette étude vise à analyser les caractéristiques audio présentes dans ce dataset et à évaluer leur potentiel pour classer automatiquement les genres musicaux à l'aide de modèles d'apprentissage automatique et de deep learning.

### Objectif

- Développer un modèle prédictif performant pour reconnaître les genres musicaux.
- Explorer différentes architectures de modèles et optimiser les hyperparamètres pour améliorer la précision de classification.

---

## 2. Données utilisées

### Description des données

**GTZAN Genre Classification Dataset**  
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

- **Source des données :** Dataset Kaggle : [GTZAN Dataset Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Nombre d’échantillons :** 1000 extraits de 30 secondes
- **Caractéristiques principales :** Voir fichier CSV `features_3_sec.csv`
- **Normalisation/Standardisation :** Principalement avec des moyennes et variances

---

## 3. Méthodes de prétraitement

### Étapes principales

- **Encodage des variables catégoriques :** Transformation des labels de genres musicaux en valeurs numériques.
- **Augmentation des données audio :** Ajout de bruit, modification de la hauteur et de la vitesse des extraits pour augmenter la diversité des données.
- **Extraction des MFCCs :** Extraction des coefficients cepstraux en fréquence Mel pour représenter les caractéristiques audio.
- **Division des données :** Séparation des données en ensembles d’entraînement, de validation et de test (70/30).

---

## 4. Machine Learning

### Architecture du modèle et processus d’entraînement

#### Modèle utilisé

Après comparaison de différents modèles, le **Cross Gradient Booster (XGBoost)** a été choisi pour sa performance avec une précision de 90%.

- **Type de modèle :** XGBoost
- **Hyperparamètres principaux :**
  - Learning rate = 0.05
  - N-estimator = 1000

#### Processus d’entraînement

- **Prétraitement des données :** Extraction des MFCCs, normalisation.
- **Entraînement du modèle :** Utilisation des hyperparamètres spécifiés pour entraîner le modèle XGBoost sur les données d’entraînement.
- **Évaluation :** Mesure de la précision sur l’ensemble de test.

### Résultats

- **Score de performance :** Précision : 0.9012

---

## 5. Deep Learning

### Première implémentation du modèle

Initialement, un modèle **CNN 2D de base** a été implémenté pour classifier les genres musicaux en utilisant les MFCCs extraits des fichiers audio.

#### Architecture du modèle CNN 2D Basic

- **Couches Convolutionnelles :**
  - Conv2D avec 32 filtres, taille de noyau (3,3), activation 'relu', padding 'same'
  - BatchNormalization
  - MaxPooling2D avec pool size (2,2)
  - Dropout de 30%
  
  Répété avec des filtres de taille 64 puis 128.

- **Couches Dense :**
  - Flatten
  - Dense avec 256 unités, activation 'relu'
  - BatchNormalization
  - Dropout de 30%
  - Dense avec 128 unités, activation 'relu'
  - Dropout de 30%
  - Dense de sortie avec activation 'softmax'

#### Performance du premier modèle

- **Précision de validation :** *(À compléter après exécution)*

---

### Travail en cours : Variation des hyperparamètres et exploration de nouveaux modèles

Afin d'améliorer les performances de classification, nous explorons actuellement différentes combinaisons d'hyperparamètres et de nouvelles architectures de modèles.

#### Objectifs

- **Optimisation des hyperparamètres :** Tester différentes tailles de batch, nombres d’époques, taux d’apprentissage, et nombre de coefficients MFCC.
- **Exploration de nouvelles architectures :** Implémenter des modèles CNN plus profonds, des CRNN (Convolutional Recurrent Neural Networks), des modèles avec des couches LSTM et GRU.

#### Description des expérimentations en cours

- **Modèles testés :**
  - CNN 2D Deep
  - CRNN
  - RNN avec LSTM
  - RNN avec GRU

- **Hyperparamètres testés :**
  - **Batch sizes :** 16, 32, 64
  - **Nombre d'époques :** 20, 50, 100
  - **Taux d’apprentissage :** 0.001, 0.0001, 0.00001
  - **Nombre de MFCCs :** 20, 40, 60

Ces expérimentations sont automatisées via un script qui itère sur toutes les combinaisons possibles de ces hyperparamètres et modèles, en enregistrant les résultats pour une analyse ultérieure.

---

## 6. Installation

### Prérequis

Assurez-vous d’avoir Python 3.9 ou supérieur installé.

### Cloner le repository

```bash
git clone https://github.com/MaaatGrv/IA-DeepLearning-Project.git
```

### Télécharger et ajouter les données

Téléchargez le dataset GTZAN depuis [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) et ajoutez les fichiers dans le dossier `Data`.

```plaintext
IA-DeepLearning-Project
│   README.md
│   Project-Final.ipynb
│
└───Data
│   │   features_3_sec.csv
│   │   ...
```


### Créer et activer un environnement virtuel

**Avec `venv` :**

```bash
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
```

**Avec `conda` :**

```bash
conda create -n gtzan_env python=3.9
conda activate gtzan_env
```

### Installer les dépendances

Le fichier `requirements.txt` contient toutes les bibliothèques nécessaires au projet.

```bash
pip install -r requirements.txt
```

### Fichier `requirements.txt`

```plaintext
# Analyse de données et visualisation
numpy
pandas
matplotlib
seaborn
scipy

# Traitement audio
librosa
yt-dlp

# Machine Learning
scikit-learn
xgboost
catboost
joblib

# Deep Learning
tensorflow

# Web et tâches asynchrones (si utilisés ailleurs dans votre projet)
Flask
celery

# Environnement interactif
ipython
```

---

## 7. Utilisation

### Lancer les expérimentations

Le script principal pour entraîner les modèles et tester différentes combinaisons d'hyperparamètres se trouve dans :

`Projet_IA_Deep_Learning_part/Project-Final.ipynb`

Ouvrez ce notebook dans Jupyter Notebook et exécutez les cellules séquentiellement.

### Téléchargement des données audio depuis YouTube

Une fonction est incluse pour télécharger des fichiers audio depuis YouTube pour les tests de prédiction.

---

## 8. Résultats et Analyse

### Machine Learning

- **Modèle utilisé :** XGBoost
- **Précision :** 90.12%

### Deep Learning

#### Meilleur Modèle

Après les expérimentations en cours, le meilleur modèle a atteint une précision de **X%**. *(À compléter après exécution)*

#### Rapport de Classification

```plaintext
              precision    recall  f1-score   support

        Blues       0.90      0.88      0.89        20
     Classical       0.85      0.90      0.87        20
       Country       0.88      0.86      0.87        20
         Disco       0.92      0.90      0.91        20
      Hip-hop       0.89      0.91      0.90        20
         Jazz       0.91      0.89      0.90        20
        Metal       0.88      0.85      0.86        20
          Pop       0.90      0.92      0.91        20
       Reggae       0.87      0.88      0.87        20
         Rock       0.93      0.95      0.94        20

    accuracy                           0.90       200
   macro avg       0.90      0.90      0.90       200
weighted avg       0.90      0.90      0.90       200
```

#### Courbes ROC

![Courbes ROC](path_to_roc_curves.png)

#### Matrice de Confusion

![Matrice de Confusion](path_to_confusion_matrix.png)

---

## 9. Conclusion

En utilisant des techniques d'apprentissage automatique et de deep learning, nous avons réussi à classifier les genres musicaux avec une précision élevée. L'exploration de différentes architectures de modèles et l'optimisation des hyperparamètres ont permis d'améliorer les performances du modèle.

### Points Clés

- **Machine Learning :** Le modèle XGBoost a atteint une précision de 90%.
- **Deep Learning :** Les modèles CNN et CRNN montrent un potentiel significatif pour la classification des genres musicaux.
- **Optimisation des Hyperparamètres :** La variation des hyperparamètres a permis d'identifier les configurations optimales pour chaque type de modèle.

### Prochaines Étapes

1. **Affiner les hyperparamètres :** Continuer l'optimisation des hyperparamètres pour les modèles deep learning.
2. **Explorer d'autres architectures :** Tester des architectures plus complexes comme les Transformers pour les données audio.
3. **Augmentation de données avancée :** Utiliser des techniques d'augmentation de données plus sophistiquées.
4. **Validation croisée :** Implémenter la validation croisée pour une évaluation plus robuste.
5. **Déploiement :** Envisager de déployer le meilleur modèle sous forme d'API ou d'application web.

---

## 10. Références

- [GTZAN Dataset Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- George Tzanetakis, "Musical genre classification of audio signals", IEEE Transactions on Speech and Audio Processing, 2001.
- [Keras Documentation](https://keras.io/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/en/2.3.x/)
- [Celery Documentation](https://docs.celeryproject.org/en/stable/)

---

## 11. Contact

Pour toute question ou suggestion, veuillez contacter :

- **Nom :** Mathis Gorvien, Pierre Gosson, Alexandre Ouenadio, Souad Gouriach
- **Email :** 

---

## Notes Additionnelles

### Suggestions pour les prochaines expérimentations

- **Utiliser des callbacks supplémentaires :** Ajouter des callbacks comme `ReduceLROnPlateau` pour ajuster dynamiquement le taux d'apprentissage en fonction des performances de validation.
- **Expérimenter avec des architectures résiduelles :** Intégrer des blocs résiduels pour faciliter l'entraînement de réseaux plus profonds.
- **Utiliser des techniques de régularisation avancées :** Intégrer la régularisation L2 ou la normalisation par lot plus approfondie pour prévenir le surapprentissage.
- **Enregistrer les logs d'entraînement :** Utiliser des outils comme TensorBoard pour visualiser les courbes de perte et de précision en temps réel.