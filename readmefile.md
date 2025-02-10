# README - Projet de Prédiction d'Âge avec YOLO et CNN

## Contexte du Projet
L’objectif de ce projet est de développer un modèle performant capable de prédire l’âge d’une personne à partir d’une image en utilisant des techniques avancées de Deep Learning. Pour cela, nous avons adopté une approche en deux étapes :

1. **Détection des visages avec YOLO** : Identification automatique des visages dans une image.
2. **Prédiction de l’âge avec un modèle CNN** : Utilisation d'un réseau de neurones convolutif (CNN) pour estimer l’âge à partir du visage détecté.

Ce projet s'appuie sur le dataset UTKFace, contenant des milliers d’images de visages étiquetées avec l'âge, le sexe et l'origine ethnique.

## 1. Détection des Visages avec YOLO
Nous avons utilisé YOLO (You Only Look Once) pour détecter les visages dans une image.

**Pourquoi YOLO ?**
- Algorithme puissant de détection d’objets en temps réel.
- Capable de détecter plusieurs visages dans une seule image.
- Fournit une localisation précise des visages, essentielle pour l’étape de prédiction d'âge.

**Entraînement du modèle YOLO**
- Annotation des images du dataset UTKFace pour entraîner YOLO à ne détecter que les visages.
- Utilisation de TensorFlow et PyTorch pour l’entraînement et la sauvegarde des poids du modèle.
- YOLO renvoie les coordonnées des visages détectés dans l’image.

## 2. Entraînement des Modèles CNN pour la Prédiction d'Âge
Une fois les visages détectés, nous avons testé plusieurs architectures CNN pour estimer l’âge des visages extraits.

**Approches Explorées :**
1. **CNN Basique** : Modèle construit à partir de zéro avec plusieurs couches convolutives.
2. **TinyVGG & ResNet50** : Modèles pré-entraînés optimisés pour la reconnaissance d’images.
3. **EfficientNetB0** : Modèle plus léger et performant pour la classification et la régression.

**Suivi des performances avec MLflow**
Nous avons utilisé MLflow pour :
- Suivre l'évolution des performances des modèles.
- Comparer plusieurs architectures.
- Sauvegarder les hyperparamètres et les résultats d’entraînement.

## 3. Sélection du Meilleur Modèle CNN
**Critères d’évaluation :**
- **MAE (Mean Absolute Error)** : Écart moyen entre la prédiction et l’âge réel.
- **MSE (Mean Squared Error)** : Mesure de la qualité des prédictions.
- **Vitesse d’inférence** : Temps nécessaire pour faire une prédiction.


## 4. Pipeline Intégrée YOLO + CNN
Nous avons conçu une pipeline complète capable de traiter une image de bout en bout :

1. **YOLO** détecte le visage et extrait les coordonnées.
2. **Le modèle CNN** prédit l’âge du visage détecté.
3. **Affichage des résultats** avec les images et les prédictions d'âge.

**Avantages de cette approche :**
- Fonctionne sur des images contenant plusieurs visages.
- Optimisée pour le temps réel.
- Modulaire : possibilité d'améliorer le modèle CNN sans modifier la détection.

## 5. Développement d'une Application Web avec Streamlit
Nous avons développé une interface utilisateur interactive pour tester notre modèle sur des images personnelles.

**Pourquoi Streamlit ?**
- Facile à utiliser avec Python.
- Permet d'afficher des images en temps réel.
- Interface utilisateur rapide à mettre en place.

**Fonctionnalités de l'application :**
- 📌 **Upload d’image** : Chargement d'une image depuis l'ordinateur.
- 📌 **Détection automatique du visage** : Identification par YOLO.
- 📌 **Prédiction de l’âge** : Estimation par le modèle CNN.
- 📌 **Affichage du résultat** : Image avec l'âge prédit affiché.

## Technologies Utilisées
- **YOLO** : Détection des visages.
- **CNN (EfficientNetB0, TinyVGG, ResNet50, CNN basique)** : Prédiction d'âge.
- **TensorFlow / Keras** : Entraînement des modèles.
- **OpenCV / NumPy** : Traitement d'images.
- **MLflow** : Suivi des expériences.
- **Streamlit** : Développement de l’interface web.



## Contribution
Les contributions sont les bienvenues ! Pour contribuer :
1. Forkez le dépôt.
2. Créez une branche.
3. Commitez vos modifications.
4. Soumettez une Pull Request.

## Licence
Ce projet est sous licence MIT.

## Remerciements
- **YOLO** pour la détection de visages.
- **TensorFlow/Keras** pour la prédiction d'âge.
- **Streamlit** pour l'application web.

