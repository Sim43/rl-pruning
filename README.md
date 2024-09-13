<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">A Reproduction of Runtime Neural Pruning</h3>

  <p align="center">
    This repository contains my reproduction of Runtime Neural Pruning.
    <br />
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#file-structure">File Structure</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the code to reproduce the experiments from the paper "Runtime Neural Pruning" by Ji Lin, Yongming Rao, Jiwen Lu and Jie Zhou, published in the 31st Conference on Neural Information Processing Systems (NIPS 2017). The paper proposes a method for dynamically pruning neural networks during inference, resulting in significant runtime improvements without sacrificing model accuracy.

[Runtime Neural Pruning](https://papers.nips.cc/paper_files/paper/2017/file/a51fb975227d6640e4fe47854476d133-Paper.pdf)

<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/rusdes/runtime-neural-pruning-reproduction.git
   ```
2. Setup (and activate) your environment
    ```sh
    conda env create -f requirements.yml
    ```
## File Structure
```
├── cnn.py
├── CONSTANTS.py
├── data
├── decision_network
│   ├── decoder.py
│   ├── encoder_change_dims.py
│   ├── encoder.py
│   ├── __init__.py
│   ├── rnn.py
│   └── train_decision_network.py
├── __init__.py
├── models
│   └── cifar10model.pth
├── README.md
├── requirements.yml
└── runtime_neural_pruning.py
```
#### `CNN.py`
This Python script trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification. The final model, which is saved in the `models/` folder, is used as a starting point for the pruning process.

#### `CONSTANTS.py`
This Python file contains all the hyperparameters that the entire framework uses. You can change any of the values in this file.

#### `data/` folder
All the datasets will be downloaded here.

#### `decision_network/decoder.py`
This file contains the Python code for the decoder part of the encoder-RNN-decoder decision network.

#### `decision_network/encoder_change_dims.py`
This file contains the Python code for changing the dimension of a vector. It is just a helper function.

#### `decision_network/encoder.py`
This file contains the Python code for the encoder part of the encoder-RNN-decoder decision network.

#### `decision_network/rnn.py`
This file contains the Python code for the RNN part of the encoder-RNN-decoder decision network.

#### `decision_network/train_decision_network.py`
This file contains the Python code for the overall training process of the decision network. It contains code for the reward calculation, state calculation, model optimization, etc.

#### `models/`
This folder stores all the saved model weights.

#### `runtime_neural_pruning.py`
This is the main python script which trains the decision network and the backbone CNN in an interleaved manner. Run the code using this script.

<!-- USAGE EXAMPLES -->
## Usage
- Step 1:
First, train a CNN network and so that the pruning script has a pretrained model to use as a starting point. Run
`python cnn.py`. This will train a CNN model on CIFAR-10 and save th emodel under the `models/` directory.

- Step 2:
All the hyperparameter values can be changed to your desired value by editing the `CONSTANTS.py` file.

- Step 3:
Next, run `python runtime_neural_network.py` to execute the RNP framework.

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/rusdes/runtime-neural-pruning-reproduction/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Improvements are always welcome! Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
