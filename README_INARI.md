# Lancer le modèle sur INARI 

## Générer une clé SSH

Sur Windows, utilisez PuTTY (Windows 7) ou OpenSSH (Windows 10). \
Attention à bien générer une clé selon le nouveau chiffrement ed25519. \

Sur macOS/OpenSSH, la commande est : ```ssh-keygen -t ed25519``` \
Avec PuTTY, la génération se fait grâce à l'utilitaire 'PuTTYgen'. \

## Connexion à INARI

Envoyez votre clé publique à Florian Leblanc avec le formulaire signé pour être ajouté sur Inari et obtenir un nom d'utilisateur. \
Connexion sous macOS/OpenSSH : ```ssh USER_NAME@inari.centre-cired.fr``` \
Connexion avec PuTTY : utilisez l'utilitaire 'PuTTY'. \

## Gurobi

Le modèle utilise le solver Gurobi, accessible gratuitement pour les chercheurs et bien plus performant que les solvers open-source. Les licenses gratuites étant nominatives, il vous faut cependant générer votre propre license et l'importer sur INARI. \
Si vous ne souhaitez pas utiliser Gurobi, vous pouvez utiliser le solver libre 'cbc'. Il vous faudra en revanche l'installer vous-même. Les performances sont sensiblement moins bonnes que Gurobi (de l'ordre d'un facteur 10). \

### Générer une license académique Gurobi

Vous pouvez générer une license académique sur le site de Gurobi : \
https://www.gurobi.com/downloads/end-user-license-agreement-academic/ \
Il faudra vous créer un compte.
Chaque license est valable un an.
Il n'y a pas de limite au nombre de licenses académiques que vous pouvez générer. \

### Importer la license sur INARI

L'importation de la license sur INARI se fait avec la commande suivante depuis INARI : \
```/data/software/gams-33.2.0-one_license/gams33.2_linux_x64_64_sfx/grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx``` \
en remplaçant par l'argument fourni lors de la création de votre license. \
Le script vous demandera où vous souhaitez stocker la license. \
La license étant nomminative, il est recommandé de la placer dans votre home repostitory (```/home/USER_NAME```).
La commande va importer un fichier ```gurobi.lic```. \

## Lancer le modèle

Commencer par indiquer le chemin vers votre license avec la variable d'environnement ```GRB_LICENSE_FILE```. \
Cette étape est importante car il existe de nombreuses licenses Gurobi sur Inari. \
Si celle-ci se trouve sur votre home directory, la commande permettant de le faire est : \
```export GRB_LICENSE_FILE=/home/USER_NAME/gurobi.lic``` \
en remplaçant USER_NAME par votre nom d'utilisateur. \
Si vous utilisez le modèle fréquemment, vous pouvez aussi l'inclure dans votre fichier ```.bashrc``` pour que la commande soit lancée à chacune de vos connexions à INARI. Néanmoins, à l'utilisation il semblerait que cette étape ne sois nécessaire qu'à la première utilisation. \

Le code est stocké sur ```/data/shared/eoles_negawatt```.
Rendez-vous y avec la commande ```cd /data/shared/eoles_negawatt```. \

Puis lancez le modèle avec : \
```nice /data/software/anaconda3/bin/pyomo solve Eoles_elec_vf_negawatt.py --solver=gurobi``` \
L'utilisation du mot clé ```nice``` est importante pour ne pas surcharger le serveur de calcul. \
Vous pouvez également modifier le solver, mais seul gurobi est pour le moment disponible sur le serveur. \
La résolution prend une centaine de secondes environ pour la version normale. \
Attention, la version 19 ans prend environ 70 minutes et utilise 16 des 32 coeurs logiques du serveur, à utiliser ponctuellement seulement. \

Si une erreur de license s'affiche (nom d'utilisateur incorrect par exemple), deux raisons possibles :
- Vous avez oublié d'indiquer le chemin vers votre license. Reprenez cette section depuis le début.
- Votre licence est expirée. Il faut en créer une autre. Reprenez toutes les étapes depuis la section 'Générer une license académique Gurobi'. \

## Quelques commandes linux utiles

La commande ```cd``` permet de se déplacer dans l'arborescence des fichiers. ```cd FOLDER``` permet d'ouvrir le dossier FOLDER, ```cd ..``` permet de revenir au dossier parent. \
La commande ```ls``` permet d'afficher la liste des fichiers présents dans le répertoire actuel. ```ls FOLDER``` permet d'afficher la liste des fichiers présents dans le répertoire ```FOLDER```. \
La commande ```cat FILE``` permet d'afficher dans la console le contenu d'un fichier simple ```FILE``` (.txt, .csv,...). \

## Gestion des fichiers

L'interface en ligne de commande n'étant pas très pratique pour exploiter les résultats, il peut être préférable de copier les résultats en local sur votre ordinateur. \

Sous macOS/OpenSSH, utilisez la commande ```scp -r USER_NAME@inari.centre-cired.fr:/data/shared/eoles_negawatt/outputs LOCAL_PATH``` dans le terminal de votre ordinateur (et pas sur INARI). \
Vous pouvez spécifier le lieu où stocker les données en remplaçant ```LOCAL_PATH``` par le chemin souhaité. \
L'argument ```-r``` permet de copier récursivement tous les sous-fichiers et sous-dossiers. Il n'est pas nécessaire dans le cas d'un fichier seul. \

Si à l'inverse vous souhaitez mettre à jour le code ou les données d'entrée, vous pouvez copier vos fichiers en inversant les deux arguments (toujours sur le terminal de votre machine) : \
```scp FILE_NAME USER_NAME@inari.centre-cired.fr:/data/shared/eoles_negawatt/...``` pour un fichier unique ; \
```scp -r FOLDER_NAME USER_NAME@inari.centre-cired.fr:/data/shared/eoles_negawatt/...``` pour un dossier. \
Pensez à modifier le chemin de manière appropriée. \

Sous Windows, il est conseillé d'utiliser WinSCP pour la gestion des fichiers. \
L'outil récupère automatiquement les infos paramétrées sur PuTTY pour fournir une interface graphique permettant de copier les fichiers dans les deux sens, pour mettre à jour le code, les données d'entrée ou pour récupérer les résultats. \
Si dans ce cas vous tentez d'écraser un fichier présent sur le serveur avec un nouveau et qu'une erreur de permission d'affiche : cliquez sur le bouton 'ignorer' et le fichier sera importé.