# Experimental Evaluation: Data and Scripts #

## General Information ##

This folder contains the data we obtained by conducting the experimental procedure described in the thesis.

It is structured as follows:
* [__BNG__](/experiments/BNG) contains the data of the experiment on BeamNG;
* [__MNIST__](/experiments/MNIST) contains the data of the experiment on MNIST;
* [__DeepJanus__](/experiments/DeepJanus) contains the data of original DeepJanus experiments.

## Dependencies ##

We ran the scripts in this folder with Python 3.11.
To easily install the dependencies with pip, we suggest to create a dedicated virtual environment and run the command:

`pip install -r requirements.txt`

Note that you will also have to add the directories `../DeepJanus-BNG` and `../DeepJanus-MNIST` to the `PYTHONPATH` environment variable.
