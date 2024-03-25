# Test Input Generator for BeamNG

## General Information
This folder contains the application of the DeepJanus approach to the steering angle prediction problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework.

## Dependencies
DeepJanus-BNG runs on Windows 10/11 (required by BeamNG simulator).

You need to have the following software installed on your machine:
- BeamNG.tech simulator v0.31.3.0\
A free version of the BeamNG simulator for research purposes can be requested at https://register.beamng.tech/
- Python 3.11
- Python libraries listed in `requirements.txt`
- DeepJanus core module

To avoid Python library conflicts, we suggest to create a dedicated virtual environment and run the command: 
```bash
pip install -r requirements.txt
pip install ../DeepJanus
```

## BeamNG Simulator Installation
1. Go to the personal link that you have been provided by email alongside with your license
2. Download BeamNG.tech.v0.31.3.0
3. Extract it to a location of your choosing
4. Place your `tech.key` file inside the `BeamNG.tech.v0.31.3.0` folder
5. Add an environment variable named `BNG_HOME` pointing to the `BeamNG.tech.v0.31.3.0` folder

## Usage

### Input
* A trained model in a format supported by Keras, the default one is in the folder `data/models`
* The seeds used for the input generation, the default ones are in the folder `../experiments/BNG/seeds`
* `deepjanus_bng/beamng_config.py` containing the configuration of the tool

### Output
When the run is finished, the tool produces the following outputs in the `data/experiments/<experiment_name>` folder:
* `config.json` reporting the configuration of the tool
* `report.json` containing the final report of the run
* the folder `archive` containing the generated individuals on the frontier (both the data structure in json and the image representation).

### Run the Tool
Run `python deepjanus_beamng.py`

### Train a New Predictor
* Run `python deepjanus_beamng.py generate-training`  to generate a new training set
* Run `python deepjanus_beamng.py train`  to train the ML model

### Generate New Seeds
Run `python deepjanus_beamng.py -s`

## Detailed Information
The project is divided as follows:

| Directory                | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| `data`                   | All input and output data for DeepJanus                      |
| `deepjanus_bng`          | Everything related to the steering angle prediction problem  |
| `deepjanus_bng/training` | Train models to be analyzed                                  |
| `levels_template`        | Custom BeamNG map that allows programmatical road generation |

### deepjanus_bng
Important files:
- `beamng_config.py`: Tool configuration for DeepJanus and problem-specific configuration
- `beamng_evaluator.py`: BeamNG simulation manager
- `beamng_interface.py`: Wrapper for all interactions with _beamngpy_

Other files are problem-specific subclasses of DeepJanus superclasses or problem-specific utilities.

### training
Important files:
- `train_dataser_recorder.py`: Dataset generator, builds random roads and simulate a vehicle driving on them, collecting data
- `train_from_recordings.py`: ML model training, builds an ML model for the steering angle prediction problem and trains it with generated data

Other files are utilities.