# SDC_Project
#### *Final project of Brainster Data Science Academy*

## Content:

#### Overview / Implementation / Results / Conclusion / References

### Overview
---
#### Intro:

The goal of our final project was to use end-to-end learning for self-driving cars, also known as Behaviour Cloning, based on Nvidia paper [1]. In order to achieve this, we used the Udacity Car Simulator powered by Unity [2]. Data was collected with manual driving the training track of the simulator using mouse, thus to achieve more smoothness of the driving trajectory. When driving run was finished the simulator generated the data that included three images and steering angle per each frame. The data then was separated into two segments: features that contained images and label that containd appropriate steering angle. After, data acquisition the next few stages were preprocessing and data augmentation, which ended with fabrication of the relevant dataset. In the training phase we constructed the Nvidia CNN model and two additional models for comparison, one based on AlexNet the other on MobileNet. Finally, in the last stage of the development we made analysis and visualization of the results, followed by autonomous driving on the challenge track by our best Nvidia model.

Training | Validation
------------|---------------
![Training](./Images/track_one.gif) | ![Validation](./Images/track_two.gif)

#### Dependencies:

The following Python libraries were utilized:

| Library | Version |
| ----------- | ----------- |
| Keras | 2.4.0 |
| TensorFlow | 2.3.0 |
| Eventlet | 0.30.2 |
| Flask | 1.1.2 |
| Sklearn | - |
| Pillow | 8.2.0 |
| Flask-SocketIO | 5.0.1 |
| Opencv | 4.0.1 |
| Pandas | 1.2.4 |
| Numpy | 1.18.5 |
| Imgaug | 0.4.0 |
| Matplotlib | 3.3.4 |

#### How to Run the Model:

This repository comes with trained model which you can directly test using the following command:

- `python drive.py model.h5`


### Implementation
---
#### Simulator Environment:

About the graphic configuration of the simulator we decided to be lower graphical quality and lower screen resolution, in our case: fastest with 800 x 600, this decision is based on research for previous users’ experience. The simulator is equipped with training mode and autonomous mode. Training mode is used for collecting data, on the other hand, autonomous mode is used to test your model performance or in other words cars behavior while driving on its own. Top view of the representative track used for autonomous driving is presented below [3]: 

Simulator Top View | 
----|
![Simulator Top View](./Images/TopView_Simulator.jpg) | 

While the car was driving in manual mode, the images from the cameras mounted on top of the car were recorded together with the steering angle for that frame, illustrated below. The data from all three cameras mounted on top of the vehicle was recorded and stored together with information about steering angle for a particulare frame.

Cameras Positions | 
----|
![Cameras Positions](./Images/Camera_Positions.jpg) | 

#### Dataset Collection:

Data collection was done while the vehicle was driving in manual mode on the representative track. Image data were acquired from the three cameras mounted on the vehicle in the simulator environment. At the end of the mouse ride, the images were stored together with the table containing information about image titles, steering angle values per each recorded frame and information about throttle, brake and speed. An example of images recorded by all the three cameras in one frame is presented below:

Images Recorded | 
----|
![Images Recorded](./Images/Cameras_View_.jpg) |

Three cameras were used for training purpose. During the data collection, time-stamped video from the cameras is captured simultaneously with the steering angle applied by the human driver. The slight difference in the field of view per each central, left, and right camera leads to a better generalization of the model. Simplified block diagram of the collection system for training data is illustrated below:

Data Collection System | 
----|
![Data Collection System](./Images/Data_Collection_System.jpg) |

