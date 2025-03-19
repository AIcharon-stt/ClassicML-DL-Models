# Classic Machine Learning and Deep Learning Models
This work provides implementations of various classic machine learning and deep learning models, including SVR, LSTM, RNN, GRU, KNN, DNN, CNN, Transformer architectures, and etc.

**Note: This repository is under active development. Some models are still being implemented and documented.**

## Overview

Understanding classic ML and DL architectures is essential for anyone working in the field of artificial intelligence. This collection serves as both a learning resource and a practical toolkit, offering:

- Standardized implementations of key algorithms
- Clear documentation explaining model architectures and principles
- Practical examples demonstrating model usage
- Performance benchmarks on standard datasets
- Hyperparameter tuning guidelines

### Traditional Machine Learning Models
- **Support Vector Regression (SVR)** - Non-linear regression using kernel methods
- **K-Nearest Neighbors (KNN)** - Classification and regression based on proximity
- **Decision Trees** - Tree-based prediction models
- **Random Forests** - Ensemble of decision trees
- **Gradient Boosting** - Sequential ensemble method for regression and classification
- **Naive Bayes** - Probabilistic classifiers based on Bayes' theorem

### Recurrent Neural Networks
- **Vanilla RNN** - Basic recurrent neural network architecture
- **Long Short-Term Memory (LSTM)** - RNN variant designed to handle long-term dependencies
- **Gated Recurrent Unit (GRU)** - Streamlined alternative to LSTM with fewer parameters
- **Bidirectional RNNs** - RNNs that process sequences in both directions

### Convolutional Neural Networks
- **Basic CNN** - Fundamental convolutional architecture
- **LeNet** - Early CNN architecture for digit recognition
- **AlexNet** - Groundbreaking CNN that popularized deep learning
- **VGG** - Deep CNN with uniform architecture
- **ResNet** - Residual network architecture enabling very deep networks

### Transformer Models
- **Vanilla Transformer** - The original attention-based architecture
- **Encoder-only Transformer** - For tasks like classification and embedding generation
- **Decoder-only Transformer** - Core architecture behind many language models
- **Transformer with various attention mechanisms** - Explorations of attention variants

### Other Deep Neural Networks
- **Multilayer Perceptron (MLP)** - Basic feedforward neural network
- **Deep Neural Networks (DNN)** - Deeper variations of MLPs
- **Autoencoders** - Unsupervised models for dimensionality reduction
- **Variational Autoencoders (VAE)** - Probabilistic autoencoders for generative modeling

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

## Usage Examples

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
