# Classic Machine Learning and Deep Learning Models

This work presents a comprehensive repository for classic machine learning and deep learning models including SVR, LSTM, RNN, GRU, KNN, DNN, CNN, Transformer architectures, and more.

⚠️ **Note: This repository is under active development. Some models are still being implemented and documented.** ⚠️

---

## Introduction

Understanding classic machine learning and deep learning architectures is essential for anyone working in the field of artificial intelligence. This collection serves as both a learning resource and a practical toolkit, offering standardized implementations, clear documentation, practical examples, performance benchmarks, and hyperparameter tuning guidelines for a wide range of fundamental algorithms.

---

## Models Included

Below is a list of models currently included (or planned) in this repository:

### Traditional Machine Learning Models

- **Support Vector Regression (SVR)**  
  [Paper](https://link.springer.com/article/10.1023/B:STCO.0000035301.49549.88) | [Original Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)  
  Non-linear regression using kernel methods.

- **K-Nearest Neighbors (KNN)**  
  [Paper](https://ieeexplore.ieee.org/document/5408784) | [Original Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)  
  Classification and regression based on proximity.

- **Decision Trees**  
  [Paper](https://link.springer.com/article/10.1007/BF00116251) | [Original Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  
  Tree-based prediction models.

- **Random Forests**  
  [Paper](https://link.springer.com/article/10.1023/A:1010933404324) | [Original Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
  Ensemble of decision trees.

- **Gradient Boosting**  
  [Paper](https://www.jstor.org/stable/2699986) | [Original Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)  
  Sequential ensemble method for regression and classification.

- **Naive Bayes**  
  [Paper](https://dl.acm.org/doi/10.1145/1015330.1015439) | [Original Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)  
  Probabilistic classifiers based on Bayes' theorem.

### Recurrent Neural Networks

- **Vanilla RNN**  
  [Paper](https://ieeexplore.ieee.org/document/58337) | [Original Implementation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)  
  Basic recurrent neural network architecture.

- **Long Short-Term Memory (LSTM)**  
  [Paper](https://ieeexplore.ieee.org/document/6795963) | [Original Implementation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)  
  RNN variant designed to handle long-term dependencies.

- **Gated Recurrent Unit (GRU)**  
  [Paper](https://arxiv.org/abs/1412.3555) | [Original Implementation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)  
  Streamlined alternative to LSTM with fewer parameters.

- **Bidirectional RNNs**  
  [Paper](https://ieeexplore.ieee.org/document/650093) | [Original Implementation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)  
  RNNs that process sequences in both directions.

### Convolutional Neural Networks

- **Basic CNN**  
  [Paper](https://ieeexplore.ieee.org/document/726791) | [Original Implementation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)  
  Fundamental convolutional architecture.

- **LeNet**  
  [Paper](https://ieeexplore.ieee.org/document/726791) | [Original Implementation](https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py)  
  Early CNN architecture for digit recognition.

- **AlexNet**  
  [Paper](https://dl.acm.org/doi/10.1145/3065386) | [Original Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)  
  Groundbreaking CNN that popularized deep learning.

- **VGG**  
  [Paper](https://arxiv.org/abs/1409.1556) | [Original Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  
  Deep CNN with uniform architecture.

- **ResNet**  
  [Paper](https://arxiv.org/abs/1512.03385) | [Original Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  
  Residual network architecture enabling very deep networks.

### Transformer Models

- **Vanilla Transformer**  
  [Paper](https://arxiv.org/abs/1706.03762) | [Original Implementation](https://github.com/tensorflow/tensor2tensor)  
  The original attention-based architecture.

- **Encoder-only Transformer**  
  [Paper](https://arxiv.org/abs/1810.04805) | [Original Implementation](https://github.com/google-research/bert)  
  For tasks like classification and embedding generation.

- **Decoder-only Transformer**  
  [Paper](https://arxiv.org/abs/2005.14165) | [Original Implementation](https://github.com/openai/gpt-2)  
  Core architecture behind many language models.

- **Transformer with various attention mechanisms**  
  [Paper](https://arxiv.org/abs/1904.02874) | [Original Implementation](https://github.com/huggingface/transformers)  
  Explorations of attention variants.

### Other Deep Neural Networks

- **Multilayer Perceptron (MLP)**  
  [Paper](https://ieeexplore.ieee.org/document/6302929) | [Original Implementation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)  
  Basic feedforward neural network.

- **Deep Neural Networks (DNN)**  
  [Paper](https://www.science.org/doi/10.1126/science.1127647) | [Original Implementation](https://www.tensorflow.org/tutorials/quickstart/beginner)  
  Deeper variations of MLPs.

- **Autoencoders**  
  [Paper](https://ieeexplore.ieee.org/document/5539957) | [Original Implementation](https://www.tensorflow.org/tutorials/generative/autoencoder)  
  Unsupervised models for dimensionality reduction.

- **Variational Autoencoders (VAE)**  
  [Paper](https://arxiv.org/abs/1312.6114) | [Original Implementation](https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py)  
  Probabilistic autoencoders for generative modeling.

### Planned Additions
We are actively working on integrating more classic and state-of-the-art models. If you have suggestions for additional models to include, feel free to open an issue or submit a pull request!

---

## Getting Started

### Prerequisites
To run the models in this repository, you will need the following dependencies:
- Python 3.7+
- PyTorch/TensorFlow
- scikit-learn
- NumPy, Pandas, Matplotlib, etc.

### Clone the repository:
```bash
git clone https://github.com/username/Classic-ML-DL-Models.git
cd Classic-ML-DL-Models
```

### Usage
Each model implementation includes detailed examples demonstrating:
1. Data preparation
2. Model initialization and configuration
3. Training procedures
4. Evaluation and testing
5. Making predictions with trained models

Here's a simplified example for LSTM:

```python
# Import the LSTM implementation
from recurrent_networks.lstm import LSTM

# Initialize the model
model = LSTM(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    dropout=0.2
)

# Train the model
model.train(X_train, y_train, epochs=100, learning_rate=0.001)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
metrics = model.evaluate(X_test, y_test)
print(f"Test MSE: {metrics['mse']}, Test MAE: {metrics['mae']}")
```

---

## Repository Structure

```
├── traditional_ml/
│   ├── svr/
│   ├── knn/
│   ├── decision_trees/
│   ├── random_forests/
│   ├── gradient_boosting/
│   └── naive_bayes/
├── recurrent_networks/
│   ├── rnn/
│   ├── lstm/
│   ├── gru/
│   └── bidirectional/
├── convolutional_networks/
│   ├── basic_cnn/
│   ├── lenet/
│   ├── alexnet/
│   ├── vgg/
│   └── resnet/
├── transformers/
│   ├── vanilla_transformer/
│   ├── encoder_only/
│   ├── decoder_only/
│   └── attention_variants/
├── deep_networks/
│   ├── mlp/
│   ├── dnn/
│   ├── autoencoders/
│   └── vae/
├── utils/
│   ├── data_preprocessing/
│   ├── visualization/
│   ├── evaluation/
│   └── hyperparameter_tuning/
├── examples/
│   ├── classification/
│   ├── regression/
│   ├── sequence_modeling/
│   └── image_processing/
└── datasets/
    ├── classification/
    ├── regression/
    ├── time_series/
    └── image/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions, suggestions, collaborations, or any copyright-related concerns, please feel free to reach out via email or open an issue on GitHub.

If you believe any content in this repository infringes upon your copyright or intellectual property rights, please contact us immediately so we can address your concerns appropriately.

Happy coding, and enjoy exploring the world of machine learning and deep learning models!
