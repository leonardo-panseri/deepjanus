# Test Input Generator for MNIST - Detailed Installation Guide #

## General Information ##
This folder contains the application of the DeepJanus approach to the handwritten digit classification problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a machine featuring an i7 processor, 16 GB of RAM, an Nvidia GeForce 940MX GPU with 2GB of memory. These instructions are for Ubuntu 18.04 (bionic) OS and python 3.6.

## Dependencies  ##
DeepJanus-MNIST runs on Ubuntu 22.04. It will likely work on other Ubuntu versions or other Linux distros, but you will have to adapt the setup process.

You need to have the following software installed on your machine:
- [CairoSVG dependencies](https://cairosvg.org/documentation/#installation) (the Python package is listed in `requirements.txt`)
- Python 3.11
- Python libraries listed in `requirements.txt`
- DeepJanus core module, it can be installed from source with `pip install ../DeepJanus`\
To easily install the dependencies, we suggest to create a dedicated virtual environment and run the command: `pip install -r requirements.txt`


## Usage ##

### Input ###
* A trained model in h5 format. The default one is in the folder `models`;
* A list of seeds used for the input generation. The default list is in the folder `original_dataset`;
* `config.py` containing the configuration of the tool selected by the user.

### Output ###
When the run is finished, the tool produces the following outputs in the folder specified by the user:
* _config.json_ reporting the configuration of the tool;
* _report.json_ containing the final report of the run;
* the folder _archive_ containing the generated inputs (both in array and image format).

### Run the Tool ###
Run the command:
`python main.py`

### Troubleshooting ###
* If tensorflow cannot be installed successfully, try to upgrade the pip version. Tensorflow cannot be installed by old versions of pip. We recommend the pip version 20.1.1.
* If the import of cairo, potrace or other modules fails, check that the correct version is installed. The correct version is reported in the file requirements.txt. The version of a module can be checked with the following command:
```
$ pip show modulename | grep Version
```
To fix the problem and install a specific version, use the following command:
```
$ pip install 'modulename==moduleversion' --force-reinstall
```
