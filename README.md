MD2D
===

Multidimensional Data to 2D. Here you can find some MATLAB scripts for t_SNE diagram visualization and some utils for source data preparation.

**Steps to use**

1. Install https://github.com/lvdmaaten/bhtsne

2. Generate 'labels.txt' and 'descriptions.txt' by means of opencvdnn utility (you need to compile QDescriptor and download neural network for description)

3. Move 'labels.txt' and 'descriptions.txt' and 't-SNE' instruments (step 1) to the 'Matlab' subfolder

4. From the MATLAB console run:

'''
    face2tsne
'''

**Popular about t-SNE**

1) https://habrahabr.ru/post/267041/

2) http://distill.pub/2016/misread-tsne/
