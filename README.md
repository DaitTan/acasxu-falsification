# Using Verification Tools to Prove Safety of a NN-based Cyber Physical System

Neural Networks are being increasingly used in the domain of perception, prediction and control of various cyber-physical systems (CPS). This has also led to an increased research in the verification and testing community to develop tools that can verify the properties of a neural network. This becomes extremely useful when deployed in systems which have a huge cost in case of a failure, aka, safety-critical systems. In this project, we utilise the ACAS Xu benchmark and try to find a single falsifying input that can falsify maximum number of neural networks from the dataset. 

This project uses [PartX](https://arxiv.org/abs/2110.10729) to find the falsifying inputs.

This project was developed for the Course Safe Autonomy for CPS by Dr. Georgios Fainekos, ASU in all 2021.

Parts of the code that involved reading ACAS-Xu benchmark neural networks and converting to tensorflow models were taken and modified from [https://github.com/rnbguy/acasxu_tf_keras](https://github.com/rnbguy/acasxu_tf_keras).

Please feel free to read the report and checkout the results. 
