# Traffic-sign-recognition

## Overview

Traffic Sign Recognition can be staged into two sections: Traffic Sign Detection and Traffic Sign Classification. In the Detection stage we aim to extract possible candidates (or regions) which contain a traffic sign (in this part, we do not care what the sign might be). In the Classification stage, we go over each Region of Interest (RoI) extracted previously and try to identify a traffic sign (which might not be in that RoI at all). The dataset.

For the dataset, images from a driving car, training and testing images for a set of signs can be found [here](https://drive.google.com/drive/u/0/folders/0B8DbLKogb5ktTW5UeWd1ZUxibDA).

The repository contains following files and folders:

- traffic_sign_recognition.pdf - Project report with detials.
- Final_TSR_detect.py - python code for running the traffic sign recognition algorithm
- dataset.pkl - pickle file containing the trained model using SVM
- templates - consists of different signs as templates, to be used for classification.

Note:

Kindly change line number 44, according to the location of the downloaded dataset from link provided in overview.
