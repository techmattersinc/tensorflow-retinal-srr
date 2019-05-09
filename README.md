# Retinal Super Resolution using Generative Adversarial Network (SRGAN)

based on 
([Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](
https://arxiv.org/abs/1609.04802))
with TensorFlow.
DataSet: [Google Diabetic retinopathy dataset](https://ai.googleblog.com/2016/11/deep-learning-for-detection-of-diabetic.html)


## Training 
![training](output/train_animated.gif)

Image shows progression of GAN over 74 epochs. GAN is able to pickup finer details as training progresses

![test](output/test_animated.gif)

Testing of the GAN using captured raw retinal imagery 

## Test Results
![Original](output/orig_retina_animated.gif)

Original Retinal imagery captured by [D-eye](https://www.d-eyecare.com/en_US/product?gclid=EAIaIQobChMI-YKO9Z-N4gIVVrbACh2VzgT1EAAYASAAEgLpw_D_BwE)

![SRR](output/srr_animated.gif)

SRR Output images

![Combined](output/combined_srr.gif)

Combined Imagery
