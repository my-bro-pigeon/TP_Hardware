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


-Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling
-
L'architecture du réseau LeNet-5 est composé de plusieurs couches :

Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST

Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

**Fichier**

Partie2.cu : 

Implémentation de la couche de convolution ainsi que la sous-echantillonnage et l'activation

On effectue un test simple : On prend une matrice d'entrée initialisée avec que des 1, un premier kernel avec un 2 au centre, un deuxième kernel avec un 1              au centre et les autre kernels à 0

Convolution 2D et subsampling
-

3.1. Layer 1 - Génération des données de test
-

3.2. Layer 2 - Convolution 2D
-

3.3. Layer 3 - Sous-échantillonnage
-

3.4. Tests
-

3.5. Fonctions d'activation
-









Partie 3 - Un peu de Python
-

entrainement de votre réseau de neurone





