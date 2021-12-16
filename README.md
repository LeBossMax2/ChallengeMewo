# Project
The goal of this project was to realised a program that automatically classify data from musical catalogue.

We designed a neuronal network solution. 

This network is implemented in [main.py](./main.py).
We also implemented F1-metrics, weigthed-F1-metrics and partial-F1-metrics and corresponding loss in this file.

# Pipeline
Our model split data in differents parts corresponding to different types of data. These parts are processed by network. Then, they are concatenated. After, that the are processed all together. At the end, these data are added to entry data. 
These steps are repeated as many times as there are blocks in our model. 

# Requirements
*train_X.csv*, *train_y.csv* and *test_X.csv* are needed in parent directory to execute properly.

Paths can be changed in [main.py](./main.py).

# Usage
Execute `python main.py`

It will train the model using *train_X.csv*, *train_y.csv* and compute the predictions corresponding to the elements in *test_X.csv*.


# Team
For this project, our team members were:
* Lucas Guédon
* Baptiste Bénard