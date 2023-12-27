![image](https://github.com/my-bro-pigeon/TP_Hardware/assets/94360349/3c82ddb8-0dd5-4b30-a26b-99495330bd4f)






Partie 1 - Prise en main de Cuda : Multiplication de matrices
-
**Objectifs :**

-Allocation de la mémoire

-Création d'une matrice sur CPU

-Affichage d'une matrice sur CPU

-Addition de deux matrices sur CPU

-Addition de deux matrices sur GPU

-Multiplication de deux matrices NxN sur CPU

-Multiplication de deux matrices NxN sur GPU

-Compléxité et temps de calcul

-Paramétrage de votre programme

**FIchier**
Matmult.cu : Réalise la multiplication de deux matrices en comparant le temps de calcul du CPU et du GPU 
tester_limites.cu : Teste la limite du GPU en faisant des calculs de multiplication de matrices de plus en plus grand jusqu'à 10k x 10k 


Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D, subsampling et activation
-
L'architecture du réseau LeNet-5 est composé de plusieurs couches :

Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST

Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

**Fichier**

Partie2.cu : 

Implémentation de la couche de convolution ainsi que la sous-echantillonnage et l'activation

On effectue un test simple : On prend une matrice d'entrée initialisée avec que des 1, un premier kernel avec un 2 au centre, un deuxième kernel avec un 1              au centre et les autre kernels à 0

On obtient bien en sortie un premier layer remplie de 0,96 (=tanh(2)) et un deuxième layer de 0,76 (=tanh(1))

Partie 3 - Modèle complet
-

3.1.












