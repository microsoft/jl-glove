<!-- # Project
- Developed by: Paidamoyo Chapfuwa , Ilker Demirel , Lorenzo Pisani, Javier Zazo, Elon Portugaly, 
H. Jabran Zahid, Julia Greissl
- Model type: Un-supervised representation learning
- License: MIT -->

# Scalable Universal T-Cell Receptor Embeddings from Adaptive Immune Repertoires (ICLR 2025)

This repository contains the Pytorch code to replicate experiments in our paper [Scalable Universal T-Cell Receptor Embeddings from Adaptive Immune Repertoires](https://openreview.net/pdf?id=wyF5vNIsO7) accepted at the International Conference on Learning Representations (ICLR 2025):

```latex
@inproceedings{
    chapfuwa2025scalable,
    title={Scalable Universal T-Cell Receptor Embeddings from Adaptive Immune Repertoires},
    author={Paidamoyo Chapfuwa and Ilker Demirel and Lorenzo Pisani and Javier Zazo and Elon Portugaly and H. Jabran Zahid and Julia Greissl},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=wyF5vNIsO7}
}
```

- Model type: Unsupervised representation learning
- License: MIT

## Model
![Model](docs/images/model_v2.png)

## Prerequisites

The code is implemented with the following dependencies:

- [Python  3.10.16](https://github.com/pyenv/pyenv)
- Additional python packages can be installed by running: 

```
poetry install
```

## Data
We consider the following public datasets:
- [Synthentic](src/jlglove/rep/_synthetic_data.py) for validating of the proposed JL-GloVe algorithm
- [ImmuneCODE](https://clients.adaptivebiotech.com/pub/covid-2020) for training the publicly available JL-GloVe TCR embeddings
- [Emerson](https://clients.adaptivebiotech.com/pub/emerson-2017-natgen) for evaluating the trained public TCR embeddings

## Model Training

* To train **JL-GloVe** embeddings using synthentic data run:

```
jl-glove generate & jl-glove train
```

## Metrics and Visualizations

* The JL-GloVe TCR embeddings derived from the 3,991 [ImmuneCODE](https://clients.adaptivebiotech.com/pub/covid-2020) repertoires are available here:

## Direct intended uses
JL-GloVe is shared for research purposes only, namely, benchmarking and inference on downstream
tasks. It is not meant to be used for clinical practice. JL-Glove was not extensively tested for
its capabilities and properties, including its accuracy and reliability in application settings,
fairness across different demographics and uses, and security and privacy.

## Out-of-scope uses
This is a research model which should not be used in any real clinical or production scenario.

## Risks and limitations
JL-GloVe TCR embeddings reflect the co-occurrence statistics of the data used for training.

## License and Usage Notices
The data, code, and model checkpoints described in this repository is provided for research use
only. The data, code, and model checkpoints is not intended for use in clinical decision-making
or for any other clinical use, and the performance of model for clinical use has not been
established. You bear sole responsibility for any use of these data, code, and model checkpoints,
including incorporation into any product intended for clinical use.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
