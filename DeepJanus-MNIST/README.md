# Test Input Generator for MNIST

## General Information
This folder contains the application of the DeepJanus approach to the handwritten digit classification problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a machine featuring an i7 processor, 16 GB of RAM, an Nvidia GeForce 940MX GPU with 2GB of memory. These instructions are for Ubuntu 18.04 (bionic) OS and python 3.6.

## Dependencies
DeepJanus-MNIST runs on Ubuntu 22.04. It will likely work on other OS, but you will have to adapt the setup process.

You need to have the following software installed on your machine:
- [CairoSVG dependencies](https://cairosvg.org/documentation/#installation) (the Python package is listed in `requirements.txt`)
- Python 3.11
- Python libraries listed in `requirements.txt`
- DeepJanus core module, it can be installed from source with `pip install ../DeepJanus`\
To easily install the dependencies, we suggest to create a dedicated virtual environment and run the command: `pip install -r requirements.txt`

## Usage

### Input
* A trained model in a format supported by Keras, the default one is in the folder `data/models`
* The seeds used for the input generation, the default ones are in the folder `data/seeds`
* `deepjanus_mnist/mnist_config.py` containing the configuration of the tool

### Output
When the run is finished, the tool produces the following outputs in the `data/experiments/<experiment_name>` folder:
* `config.json` reporting the configuration of the tool
* `report.json` containing the final report of the run
* the folder `archive` containing the generated individuals on the frontier (both the data structure in json and the image representation in svg format).

### Run the Tool
Run `python deepjanus_mnist.py`

### Train a New Predictor
Run `python deepjanus_mnist.py train`  to train the ML model

### Generate New Seeds
Run `python deepjanus_mnist.py -s`

## Detailed Information
The project is divided as follows:

| Directory                | Description                                         |
|--------------------------|-----------------------------------------------------|
| `data`                   | All input and output data for DeepJanus             |
| `deepjanus_mnist`        | Everything related to the digit recognition problem |

### deepjanus_bng
Important files:
- `mnist_config.py`: Tool configuration for DeepJanus and problem-specific configuration
- `model_trainer.py`: ML model training, builds an ML model for the digit recognition problem and trains it with the MNIST dataset
- `seed_generator.py`: Seed generator, creates DeepJanus-MNIST seeds for a trained model

Other files are problem-specific subclasses of DeepJanus superclasses or problem-specific utilities.
