![image](https://github.com/my-bro-pigeon/TP_Hardware/assets/94360349/3c82ddb8-0dd5-4b30-a26b-99495330bd4f)






<center><h1> Avant Propos </h1></center>

_L'objectif de ce projet est d'implémenter un CNN classique : LeNet-5 en utilisant Cuda. Cela permet d'appréhender la parallélisation des calculs via l'utilisation de GPU NVidia_

Partie 1 - Prise en main de Cuda : Multiplication de matrices
-
## **Objectifs :** 🎯

Allocation de la mémoire // Création d'une matrice sur CPU // Affichage d'une matrice sur CPU // Addition de deux matrices sur CPU // Addition de deux matrices sur GPU-Multiplication de deux matrices NxN sur CPU // Multiplication de deux matrices NxN sur GPU // Compléxité et temps de calcul // Paramétrage de votre programme

## **Fichier** 📁

-> _Matmult.cu_ : Réalise la multiplication de deux matrices en comparant le temps de calcul du CPU et du GPU 

-> _tester_limites.cu_ : Teste la limite du GPU en faisant des calculs de multiplication de matrices de plus en plus grand jusqu'à 10k x 10k 

_L'affichage d'une matrice N x N x 3 se fait comme suit : matrice 1 : N x N , matrice 2 : N x N , Matrice 3 : N x N_

Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D, subsampling et activation
-
## **Objectifs :** 🎯

Implémentation des premières couches de l'architecture du réseau LeNet-5 :

Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST

Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

## **Fichier** 📁

-> _Partie2.cu_ : 

Implémentation de la couche de convolution ainsi que la sous-echantillonnage et l'activation

On effectue un test simple : On prend une matrice d'entrée initialisée avec que des 1, un premier kernel avec un 2 au centre, un deuxième kernel avec un 1              au centre et les autre kernels à 0

On obtient bien en sortie un premier layer remplie de 0,96 (=tanh(2)) et un deuxième layer de 0,76 (=tanh(1))

Pour l'ensemble de ces fichiers et des fichiers à venir, nous choisi comme architecture Grid/Block comme ceci 

<img width="200" alt="image" src="https://github.com/my-bro-pigeon/TP_Hardware/assets/81351824/27fb65ab-3ab7-4535-8359-2c4f2c043f66">

Chaque élément de notre matrice de sortie correspond à un block dans une représentation 3d de la forme de la matrice de sortie. Seul un thread par block est utilisé.


Partie 3 - Modèle complet
-
## **Objectifs :** 🎯

Implémentation de toutes les couches du model // Importation du dataset MNIST // Exportation des poids du model

## **Fichier** 📁

-> _Partie3.cu_ : implémenation du modèle complet, ajout de la couche de convolution 2, du flatten et des trois couches dense

-> _LeNet5.ipynb_ : récuperation des poids du modèle entrainé (grâce au fichier généré FashionMNIST_weights.h5) 

<img width="304" alt="image" src="https://github.com/my-bro-pigeon/TP_Hardware/assets/81351824/8d8ef97b-f308-4739-afee-e487f84fd457">


-> _full_model.cu_ : importation des poids dans notre modèle grâce aux fichiers .h contenus dans le fichier /weights et ajout de la fonction de convolution 3D utile pour la deuxième couche de convolution de notre modèle. 

Résultat pour un "1" en entrée : 

<img width="470" alt="image" src="https://github.com/my-bro-pigeon/TP_Hardware/assets/81351824/203f6491-03f1-4ee2-8cd6-ab1294627b07">

Les résultats ne sont pas satisfaisants, beaucoup d'erreurs.  

-> _model_V2.cu_ : Dans _full_model.cu_ nous utilisons des poids issus d'un modèle avec des biais sur les couches de convolution mais sans utiliser les biais des ces couches. On réentraine donc le modèle sans biais sur les couches de convolution <img width="583" alt="image" src="https://github.com/my-bro-pigeon/TP_Hardware/assets/81351824/aac0c670-c13d-4817-be2b-2e7b3793e401">

On utilise donc ici les fichiers .h de dossier /weights_nobias. Nous avons également remplacé les "float" par des "double".
Cependant les résultats ne sont pas forcément mieux qu'avec la première version du modèle. 













