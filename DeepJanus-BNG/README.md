# Test Input Generator for BeamNG #

## General Information ##
This folder contains the application of the DeepJanus approach to the steering angle prediction problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a Windows machine equipped with an i9 processor, 32 GB of memory, and an Nvidia GPU GeForce RTX 2080 Ti with 11GB of dedicated memory.

## Dependencies ##
You need to have the following software installed on your machine:
- BeamNG.tech simulator v0.30.5.0\
A free version of the BeamNG simulator for research purposes can be found at https://register.beamng.tech/
- Python 3.11
- Python libraries listed in `requirements.txt`\
To easily install the dependencies, we suggest to create a dedicated virtual environment and run the command: `pip install -r requirements.txt`



## BeamNG Simulator Installation ##
1. Go to the personal link that you have been provided by email alongside with your license
2. Download BeamNG.tech.v0.30.5.0
3. Extract it to a location of your choosing
4. Place your `tech.key` file inside the `BeamNG.tech.v0.30.5.0` folder
5. Add an environment variable named `BNG_HOME` pointing to the `BeamNG.tech.v0.30.5.0` folder

The less powerful graphics card we have successfully tested our tool with is an NVIDIA GeForce 940MX.

## Usage ##

### Input ###

* A trained model in a format supported by Keras, the default one is in the folder `data/trained_models_colab`
* The seeds used for the input generation, the default ones are in the folder `data/member_seeds`
* `core/config.py` and `self_driving/beamng_config.py` containing the configuration of the tool

### Output ###
When the run is finished, the tool produces the following outputs in the `data/experiments/<experiment_name>` folder:
* `config.json` reporting the configuration of the tool
* `report.json` containing the final report of the run
* the folder `archive` containing the generated individuals on the frontier (both the data structure in json and the image representation in svg format).

### Run the Tool ###
Run `python deepjanus_beamng.py`

### Train a New Predictor ###
* Run `python deepjanus_beamng.py generate-training`  to generate a new training set;
* Run `python deepjanus_beamng.py train`  to train the ML model.

### Generate New Seeds ###
Run `python deepjanus_beamng.py -s`
