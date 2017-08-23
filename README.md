
# Self-Driving Car Engineer Nanodegree


## Project 3: **Behavioral Cloning** 

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

***

[//]: # (Image References)

[image1]: ./images_md/Nvidia_model.png "Nvidia model"

### Overview

My project contains the following files:
- model.py: data preprocessing, training model, generator for data and running the training
- drive.py: as provided with adjusted speed
- model.h5: trained model
- README.md: this file

### Approach

#### Training model

At the beginning of this project I did some research and found the [Nvidia End-to-end-model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) including a working CNN model already used for autonomous driving in vehicles. I decided to also go with this model and implemented into my model.py (line 11). 

![alt text][image1]

Instead of the proposed image size of 66x200 I used the the size recorded from the simulator 160x320. The image is cropped to the relevant region and normalized. Then following 4 Convolutional Layers followd by 3 Fully-connected Layers. In between these 3 Layers I put Dropuout Layers since I made good experience in the 2nd project. 

#### Handling data

Since this projects goal was to have the car driving autonomously by learning with a neural network it was clear that the project might have a huge focus on data. The data provided from udacity was ok to have the car run, but it wasn't satisfying at all. 

I used a generator (line 61 model.py) to hand over random data to the training model. While loading an image (line 44 model.py) it is chosen randomly whether it is a center, right or left image. For right and left image there is an angle correction of |0.2|, determined empirically. In case the generator is creating a batch of training data there is some image augmentation added. With a chance of 50% each the image is fliped horizontally or/and adjusted regarding its brightness. This is to even provide more data to the training model.

#### Recording additional data

Since the provided dataset was not big enough to have the car run smoothly and the simulator was provided to record, what seemed to be fun, I generated additional data. For the purpose of smoother steering with an axis and to avoid the binary steering with the keybord I drove the tracks with a PS4 controller. This worked well plug and play.

I did 4 additional datasets with 2 rounds around the course each:
- Track 1 clockwise
- Track 1 counter-clockwise
- Track 2 clockwise
- Track 2 counter-clockwise

Adding this data to the project data provided by udacity I ended up with 18948 datapoints.

#### Training

The training was done on an AWS with the combined dataset. The batch size and the number of epochs was chosen empirically during some runs. Finally I trained with a batch size of 128 in 19 epochs. The data was split into 80% training data and 20% validation data and ended up with a loss of 0.031 (training) and 0.025 (validation). 

#### Running the model in simulator 

I started an instance of the simulator using the provided code to have the vehicle autonomously driving using the generated model.h5 on both tracks:
```python
python drive.py model.h5
```
When I was satisfied with the result I started the instance again recording the images and generating a video afterwards. 

My model was able to drive both tracks with an increased speed (set to 20) without leaving the road. The videos for both tracks can be found on Youtube: [Track 1](https://youtu.be/5BFTcZ-a0_A) and [Track 2](https://youtu.be/BKgxwA-sVZQ).


### Discussion

I am really happy with the result running both tracks on a solid path. As mentioned before this project was all about the data and there I do see the shortcomings in my pipeline. There is room for even more augmentation to provide even more data generalizing the model.

There was a time when I was stuck on the code, but this was more about my lack of python knowledge then the algorithm as is. In the end I learned a lot and the code worked driving both of the tracks. 




