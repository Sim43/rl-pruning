<br />
<p align="center">
  <h3 align="center">A Reproduction of Runtime Neural Pruning</h3>

  <p align="center">
    This repository contains a reproduction of <a href="https://papers.nips.cc/paper_files/paper/2017/file/a51fb975227d6640e4fe47854476d133-Paper.pdf" target="_blank">
      Runtime Neural Pruning (NeurIPS 2017)
    </a>, aimed at reducing the computational cost of CNNs during inference by dynamically pruning convolutional neural networks.
    <br />
    <br />
  </p>
</p>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#key-features">Key Features</a>
    </li>
    <li>
      <a href="#how-it-works">How it Works</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li><a href="#file-structure">File Structure</a></li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li>
          <a href="#running-the-runtime-neural-pruning-pipeline">Running the Runtime Neural Pruning Pipeline</a>
        </li>
        <li>
          <a href="#available-arguments">Available Arguments</a>
        </li>
      </ul>
    </li>
    <li><a href="#understanding-the-output">Understanding the Output</a></li>
    <li><a href="#costumizing-the-code">Customizing the Code</a></li>
  </ol>
</details>

## About The Project

This repository contains the code to reproduce the experiments from the paper [Runtime Neural Pruning (NeurIPS 2017)](https://papers.nips.cc/paper_files/paper/2017/file/a51fb975227d6640e4fe47854476d133-Paper.pdf) by Ji Lin, Yongming Rao, Jiwen Lu and Jie Zhou, published in the 31st Conference on Neural Information Processing Systems (NIPS 2017). The paper proposes a method for dynamically pruning neural networks during inference, resulting in significant runtime improvements without sacrificing model accuracy. Since no public code was available for this work, I have made the full code publicly accessible to facilitate reproducibility and further research in neural network pruning.

## Key Features
- **CNN Pruning via Reinforcement Learning**: Dynamically prunes CNN layers during inference, depending on how complex the input image is.  
- **Alternate Training of the decision network and the CNN**: Interleaves pruning with model fine-tuning for optimal accuracy.  
- **Performance Comparison**: Compares original vs. pruned model performance and number of multiplications performed.  
- **Reproducibility**: Reimplemented the approach from the original NeurIPS paper on neural network pruning using reinforcement learning, and made the code publicly available since no public code existed.

## How it Works
Runtime Neural Pruning (RNP) is a framework designed to dynamically prune convolutional kernels in a CNN during inference, reducing computational cost while maintaining performance. It consists of two key components:

1. **Backbone CNN**: A standard CNN architecture that processes input images and produces feature maps.
2. **Decision Network**: A reinforcement learning-based network that decides which convolutional kernels to prune based on the input image and intermediate feature maps.

The process works as follows:

1. The **Decision Network** uses a **Markov Decision Process (MDP)** to make pruning decisions layer-by-layer. It encodes feature maps, aggregates information across layers using an RNN, and outputs Q-values for pruning actions.
2. Pruning actions are defined incrementally, grouping output channels and calculating only the necessary feature maps based on the **input's complexity**.
3. The framework is trained using **Q-learning**, where rewards balance classification accuracy and computational efficiency. The backbone CNN and decision network are **trained alternately** to optimize performance.

During inference, RNP **dynamically** adjusts pruning based on the input, enabling efficient inference without retraining. This allows the same model to adapt to different computational constraints.

## Installation
Before running the script, make sure you have all dependencies installed. You can set up your environment using the provided `environment.yml` file:
1. Clone the repo
   ```sh
   git clone https://github.com/rusdes/runtime-neural-pruning-reproduction.git
   ```
2. Setup and activate your environment
    ```sh
    conda env create -f requirements.yml
    conda activate runtime_neural_pruning
    ```
## File Structure
```
├── backbone_cnn
│   ├── cnn.py                    # Defines the base CNN architecture
│   └── cnn_train.py              # Trains the backbone CNN model
├── CONSTANTS.py                  # Stores global constants and hyperparameters
├── data                          # Directory for dataset storage
├── decision_network
│   ├── cnn_finetuner.py          # Handles fine-tuning of a CNN model with runtime neural pruning
│   ├── dqn.py                    # Defines the Deep Q-Network (DQN) model with an encoder-RNN-decoder architecture
│   ├── dqn_trainer.py            # Trainer for the DQN model
│   ├── __init__.py
│   └── pruner_manager.py         # Manages pruning operations for the CNN model using the trained policy network.
├── driver.py                     # Main script to execute the full pipeline
├── environment.yml               # Environment dependencies
├── __init__.py
├── models                        # Directory to store trained models
├── README.md
└── utils.py                      # Helper functions
```
## Usage
### Running the Runtime Neural Pruning Pipeline
To execute the Runtime Neural Pruning (RNP) pipeline, simply run:

```sh
    python driver.py --dataset FashionMNIST
```
### Available Arguments
| Argument               | Default        | Description                                      |
|------------------------|---------------|--------------------------------------------------|
| `--dataset`           | `FashionMNIST` | Choose dataset: `FashionMNIST` or `CIFAR10`     |
| `--seed`              | `42`           | Random seed for reproducibility                 |
| `--num_episodes_rl`   | `100`          | Number of episodes for RL training              |
| `--finetune_steps_cnn`| `50`           | Number of fine-tuning steps for CNN             |

Example with custom parameters:
```sh
    python driver.py --dataset FashionMNIST --seed 123 --num_episodes_rl 500 --finetune_steps_cnn 3000
```
## Understanding the Output
Once the script runs successfully, you will see a comparison of the original and pruned models, including:
- Number of multiplications per convolutional layer
- Total multiplications in the model
- Test accuracy before and after pruning

### Example Output
```text
Comparison of Multiplications for processing one image:
╒══════════════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╕
│                  │   conv1   │   conv2   │   conv3   │   conv4   │   Total   │
╞══════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
│ Original Model   │  451,584  │ 57,802,752│ 57,802,752│ 14,450,688│130,507,776│
├──────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ Pruned Model     │  451,584  │ 46,061,568| 34,320,384│  2,709,504│ 83,543,040│
╘══════════════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╛

╒══════════════════╤═══════════════╕
│                  │ Test Accuracy │
╞══════════════════╪═══════════════╡
│ Original Model   │         90.12 │
├──────────────────┼───────────────┤
│ Pruned Model     │         88.26 │
╘══════════════════╧═══════════════╛

Pruned model contains 64.01% of the original model's multiplications.
```
## Customizing the Code
- Modify `CONSTANTS.py` to tweak hyperparameters.
- If you want to experiment with different model architectures, edit `backbone_cnn/cnn.py` 