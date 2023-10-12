# Eoles

Eoles est un modèle d'optimisation de l'investissement et de l'exploitation du système énergétique en France cherchant à minimiser le coût total tout en satisfaisant la demande en énergie. \
Voici une présentation d'une version antérieure du modèle : _http://www.centre-cired.fr/quel-mix-electrique-optimal-en-france-en-2050/_ \
Ce Gitlab est la traduction en python du modèle Eoles_elec_pro intialement en GAMS. \
Vous pouvez le retrouver ici : _https://github.com/BehrangShirizadeh/EOLES_elec_pro_

---

### Lancer le modèle avec Pyomo

---

#### **Installation des dépendances**

Pour pouvoir lancer le modèle vous aurez besoin d'installer certaines dépendances dont ce programme à besoin pour fonctionner :

* **Python** :
Python est un langage de programmation interprété, utilisé avec Pyomo il va permettre de modéliser Eoles. \
Vous pouvez télécharger la dernière version sur le site dédié : *https://www.python.org/downloads/* \
Ensuite il vous suffit de l'installer sur votre ordinateur. \
Si vous comptez installer Conda ou si vous avez installé Condé sur votre ordinateur, Python à des chances d'être déjà installé.

* **Conda** ou **Pip** selon votre préférence :
Conda et Pip sont des gestionnaires de paquets pour Python.
    * **Conda** \
    Vous pouvez retrouver toutes les informations nécéssaires pour télécharger et installer Conda ici: \
    _https://docs.conda.io/projects/conda/en/latest/user-guide/install/_ \
    __Attention à bien choisir la version de Conda en accord avec la version de Python !__ \
    Vous pouvez installer Miniconda qui est une version minimale de Conda,\
    cela vous permettra de ne pas installer tous les paquets compris dans Conda, \
    mais seulement ceux qui sont nécéssaires.
    * **Pip** \
    Vous pouvez retrouver toutes les informations nécéssaires pour télécharger et installer Pip  ici : \
    _https://pip.pypa.io/en/stable/installing/_ \
    Pip est également installé si vous avez installé Conda.

* **Pandas** :
Pandas est une librairie de Python qui permet de manipuler et analyser des données facilement. \
Pandas est open-source.
Ouvrez une interface de commande et tapez ceci : \
```conda install pandas```, avec Conda \
```pip install pandas```, avec Pip \
Vous pouvez retrouver plus d'information ici : _https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html_

* **Pyomo** :
Pyomo est un langage de modélisation d'optimisation basé sur le langage Python. \
Pyomo est open-source.
Ouvrez une interface de commande et tapez ceci : \
```conda install -c conda-forge pyomo```, avec Conda \
```pip install pyomo```, avec Pip \
Vous pouvez retrouver plus d'information ici : _https://pyomo.readthedocs.io/en/stable/installation.html_

* **Solveur** :
Le solveur que ce modèle utilise est le solveur Gurobi, bien plus rapide que le solveur Cbc. \
Des licenses gratuites sont mises à dispositions pour les chercheurs et étudiants.
Pour utiliser Gurobi :
    * Se créer un compte et télécharger Gurobi Optimizer ici : _https://www.gurobi.com/downloads/_
    * Demander une license académique gratuite : _https://www.gurobi.com/downloads/end-user-license-agreement-academic/_
    * Utiliser la commande ```grbgetkey``` pour importer sa license, comme indiquer sur celle-ci.

#### **Récupération du code :**

Si vous n'avez pas installé Git sur votre ordinateur, vous pouvez téléchargez le dossier sur ce GitLab, dans le format que vous souhaitez.
Sinon, vous pouvez récupérer les fichiers de ce GitLab à travers la commande :\
```git clone https://gitlab.in2p3.fr/quentin.bustarret/eoles.git```\
Un dossier sera créé dans le répertoire courant avec tout les fichiers contenus dans ce GitLab. \

#### **Lancement du code :**

Continuez à travers une interface de commande, déplacez-vous dans le répertoire téléchargé :\
```cd eoles``` \
*cd qui signifie change directory, permet de se déplacer dans vos dossiers à travers les lignes de commandes* \
Puis lancer la résolution du modèle avec la commande :\
```python3 Eoles_elec_vf_preprocess.py``` (version simplifiée - RECOMMANDÉE)\ 
```python3 RUN.py``` (version complète avec corrections pour le nucléaire)\
_Attention, sur Windows, il se peut que la commande fonctionne uniquement dans une interface de commande Anaconda (Anaconda Prompt)_

---

### Modifier les données d'entrées

---

Vous pouvez observer les données d'entrées dans le dossier **inputs**, \
Pour les modifier, faites attention à respecter le format des données : \
Pensez à bien mettre les noms des technologies en accord avec celles présentes dans le fichier .py. \
La casse est importante, c'est-à-dire que les variables en minuscule doivent rester en minuscule et inversement. \
Sinon cela pourrait causer des erreurs. \
Concernant les nombres, faites attention à séparer la partie entière de la partie décimale avec un point et non une virgule. \
Cela pourrait aussi causer des erreurs au lancement du programme.
Vous pouvez retrouver plus d'informations concernant les données d'entrées dans le fichier __inputs.txt__

---

### Fichiers de Sorties

---

Le programme sort 4 fichiers: \
**Summary** : Petit résumé du programme, il contient l'objectif final (le coût) mais aussi d'autres informations comme l'écrêtement ou les pertes de stockage. \
**Hourly Generation** : Contient la génération horaire pour chaque technologie ainsi que d'autres informations heure par heure. \
**Elec_Balance** : Contient le bilan électrique, c'est à dire la production et la consommation des résultats. \
**Capacities** : Contient les capacités en énergie et en puissance des technologies. \
Pour un meilleur affichage sur Excel, Vous pouvez faire : \
Selectionner la colonne A > Aller dans le menu : Données > Convertir > Selectionner Délimité > Suivant > Cocher la case virgule > Terminer.

---

### Divers

---

Si vous avez une question sur le programme ou bien des problèmes d'installation, vous pouvez les poser à l'adresse : \
_quentin@bustarret.com_ (Quentin Bustarret)
