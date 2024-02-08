# Model-based Exploration of the Frontier of Behaviours for Deep Learning System Testing

## General Information ##
This repository contains the tools and the data of the paper "Model-based Exploration of the Frontier of Behaviours for Deep Learning System Testing"
 by V. Riccio and P. Tonella, published in the [Proceedings of the ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 2020)](https://dl.acm.org/doi/abs/10.1145/3368089.3409730).

## Repository Structure ##
The project is structured as follows:

* [__DeepJanus__](/DeepJanus) contains the DeepJanus core package, that provides a way to easily adapt DeepJanus to specific problems;
* [__DeepJanus-BNG__](/DeepJanus-BNG) contains the DeepJanus tool adapted to the self-driving car case study;
* [__DeepJanus-MNIST__](/DeepJanus-MNIST) contains the DeepJanus tool adapted to the handwritten digit classification case study;
* [__experiments__](/experiments) contains the raw experimental data reported in the paper and the scripts to obtain the data.

_Note:_ each sub-package contains further specific instructions.

## Reference

If you use our work, please cite DeepJanus in your publications. 
Here is an example BibTeX entry:

```
@inproceedings{RiccioTonella_FSE_2020,
	title= {Model-based Exploration of the Frontier of Behaviours for Deep Learning System Testing},
	author= {Vincenzo Riccio and Paolo Tonella},
	booktitle= {Proceedings of the ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
	series= {ESEC/FSE '20},
	publisher= {Association for Computing Machinery},
	doi = {10.1145/3368089.3409730},
	pages= {13 pages},
	year= {2020}
}
```

## License ##
The software we developed is distributed under MIT license. See the [license](/LICENSE) file.

## Contacts

For any related question, please contact Vincenzo Riccio ([vincenzo.riccio@usi.ch](mailto:vincenzo.riccio@usi.ch)) 
or Paolo Tonella ([paolo.tonella@usi.ch](mailto:paolo.tonella@usi.ch)).
