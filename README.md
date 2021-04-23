# SDC_Project
##### *Final project of Brainster Data Science Academy*

## Content:

#### Overview / Implementation / Results / Conclusion / References
---

### Overview
---
#### Intro:

The goal of our final project was to use end-to-end learning for self-driving cars, also known as Behaviour Cloning, based on Nvidia paper [1]. In order to achieve this, we used the Udacity Car Simulator powered by Unity [2]. Data was collected with manual driving the training track of the simulator using mouse, thus to achieve more smoothness of the driving trajectory. When driving run was finished the simulator generated the data that included three images and steering angle per each frame. The data then was separated into two segments: features that contained images and label that containd appropriate steering angle. After, data acquisition the next few stages were preprocessing and data augmentation, which ended with fabrication of the relevant dataset. In the training phase we constructed the Nvidia CNN model and two additional models for comparison, one based on AlexNet the other on MobileNet. Finally, in the last stage of the development we made analysis and visualization of the results, followed by autonomous driving on the challenge track by our best Nvidia model.

Training | Validation
------------|---------------
![Training Image](./Images/track_one.gif) | ![Validation Image](./Images/track_two.gif)

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

---

 
