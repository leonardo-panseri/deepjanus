# Probabilistic DeepJanus

## General Information
This repository contains the tools and the data of master's degree thesis "Automated detection of probabilistic behavior 
boundaries of ML-enabled autonomous systems" by Leonardo Panseri at Politecnico di Milano.

This project is based on previous research conducted by V. Riccio and P. Tonella at Università della Svizzera Italiana.
You can read more in the paper "Model-based Exploration of the Frontier of Behaviours for Deep Learning System Testing",
published in the [Proceedings of the ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 2020)](https://dl.acm.org/doi/abs/10.1145/3368089.3409730).

The goal of this project is to characterize in a probabilistic way inputs of an autonomous system that may cause problems.
This is done by modeling the input domain and exploring it with a genetic algorithm.

## Repository Structure
The project is structured as follows:

* [__DeepJanus__](/DeepJanus) contains the DeepJanus core package that provides a way to easily adapt DeepJanus to specific problems;
* [__DeepJanus-BNG__](/DeepJanus-BNG) contains the DeepJanus tool adapted to the self-driving car case study;
* [__DeepJanus-MNIST__](/DeepJanus-MNIST) contains the DeepJanus tool adapted to the handwritten digit classification case study;
* [__experiments__](/experiments) contains the raw experimental data reported in the paper and the scripts to obtain the data.

_Note:_ each sub-package contains further specific instructions.

## License
The software we developed is distributed under MIT license. See the [license](/LICENSE) file.
