# DL_Spring_2019 國立交通大學電機工程學系，深度學習
Deep Learning Spring 2019 by Jen-Tzung Chien@ ECE NCTU Taiwan 授課教師：簡仁宗
(PS. This is my first time taking a Deep Learning class, the code might not be optimal, but I have tried my best to make it readable and clean)

Course website [here](https://plus.nctu.edu.tw/courses/34767) 課程網站 [點此](https://plus.nctu.edu.tw/courses/34767)

## HW lists and details 作業解答以及各作業內容
HW score shown as below
![](https://i.imgur.com/ubRvMs5.png)

* HW1, Score 98/100
    * [Spec PDF](https://github.com/Alfons0329/DL_Spring_2019/blob/master/HW1/dl_hw1.pdf)
    * Handcraft a DNN classifier from scratch, without any assistence from python module, i.e. no import Keras, torch nor tenserflow.
    * An 0-1 classifier from the famous [titanic dataset in Kaggle](https://www.kaggle.com/c/titanic)
    * Analyze the correlation-coeffiecnt and find out the principal column element.
    * Discuss the need of one-hot encoding.
    * [LaTeX report](https://github.com/Alfons0329/DL_Spring_2019/blob/master/HW1/HW1_Report_0416324.pdf)
    * [LaTeX report source code](https://github.com/Alfons0329/DL_Spring_2019/tree/master/report_latex/HW1)
* HW2, Score 100/100
    * [Spec PDF](https://github.com/Alfons0329/DL_Spring_2019/tree/master/HW2/HW2.pdf)
    * CNN 
        * A traditional CNN classifier to classify the image of various types of animals (CUDA and NVIDIA GPU is required for accelerating the tasks.)
        * Background knowledge with image processing might help this homework.
        * PyTorch is allowed and used in this homework, but auto model build framework such as autoML is not allowed.
    * RNN / LSTM
        * Use RNN / LSTM for text analyzing and judge whether a paper with certain title will get accepted or not.
        * `Word vector` is important, which is the fundamental of NLP
        * Compare the performance b/w RNN and LSTM as well as discussing the reason behind it.
        * Try explain the reason of `Gradient Vanishing` or `Gradient Exploding`
    * [LaTeX report](https://github.com/Alfons0329/DL_Spring_2019/blob/master/HW2/HW2_Report_0416324.pdf)
    * [LaTeX report source code](https://github.com/Alfons0329/DL_Spring_2019/tree/master/report_latex/HW2)
 
* HW3, Score 95/100
    * [Spec PDF](https://github.com/Alfons0329/DL_Spring_2019/tree/master/HW3/hw3.pdf)
    * Variational Autoencoder
        * Use VAE to compress images and resconstruct them.
        * Background knowledge with probabilities and statistics might help this homework.
    * CycleGAN
        * [Implement this paper](https://github.com/junyanz/CycleGAN)
        * Use CycleGAN to fake generate some anime/cartoon character from the other style. 
    * [LaTeX report](https://github.com/Alfons0329/DL_Spring_2019/blob/master/HW3/HW3_Report_0416324.pdf)
    * [LaTeX report source code](https://github.com/Alfons0329/DL_Spring_2019/tree/master/report_latex/HW3)

## Final Project 期末專題
* A `Style Transfer` to make picture of someone younger / older.
* The original idea is taken from [here].(https://www.pytorchtutorial.com/pytorch-style-transfer/)
* Our team proposed 2 ways to implement, the CycleGAN from above and Neural Style Transfer from [here](https://arxiv.org/pdf/1705.04058.pdf%20http://arxiv.org/abs/1705.04058.pdf).
* [LaTeX report](https://github.com/Alfons0329/DL_Spring_2019/blob/master/Final_Project/Group13_Final_Project_Report.pdf)
* [LaTeX report source code](https://github.com/Alfons0329/DL_Spring_2019/tree/master/report_latex/Final_Project)
