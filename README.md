```markdown
# Higher Education Students Performance Evaluation

Les données ont été recueillies auprès des étudiants de la Faculté d'ingénierie et de la Faculté des sciences de l'éducation en 2019. L'objectif est de prédire les performances des étudiants à la fin du semestre à l'aide de techniques d'apprentissage automatique.


## 🚀 Prérequis

- **Python** : version 3.10.13  
  > TensorFlow 2.18 nécessite Python ≤ 3.12. Vérifiez avec `python --version`.

## 📦 Installation

1. **Clonez le dépôt**  
   ```bash
   git clone https://github.com/Pambawl23/Projet_1_MLearning.git
   cd votre-projet
   ```

2. **Créez et activez un environnement virtuel**  
   ```bash
   python -m venv env
   # ou .\env\Scripts\activate     # Sur Windows
   ```
3. **Installez les dépendances**  
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Utilisation

Lancez le script principal :

```bash
python algorithme.py #Entrainement du modele
python app.py        #Lancement de l'application(Interface)
```

## 📁 Structure du projet (exemple)

```
.
├── requirements.txt       # Dépendances
├── README.md              # Ce fichier
├── dataset.py             # Données d'entraînement
├── algoritme.py           # Modèles
└── app.py                 # Application(Interface)
```
