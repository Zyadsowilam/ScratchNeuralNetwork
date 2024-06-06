# ScratchNeuralNetwork
This repo is a self-learning projects of how to apply neural network using Only numpy
# MNIST Dataset

The MNIST database contains handwritten digits, commonly used for training various image processing systems.

## Dataset Overview

Each image in the MNIST dataset is 28x28 pixels, where the pixel values range from 0 to 255, representing black and white. The images are flattened to 784 (28x28) pixels to be used as input to a neural network.

## Neural Network Architecture

### Input Layer

- **Input Layer**: Each row represents one image. By taking the transpose of the image matrix, each column represents one image. The input layer size is 784, corresponding to the 784 pixels of each image.

### Hidden Layer

- **Hidden Layer**: The hidden layer has 10 neurons. The input to the hidden layer is calculated as follows:
  \[
  \text{hidden\_layer}[1](10 \times m) = \text{Weight}[1](10 \times 784) \times \text{input\_layer}[0] + \text{bias}[1](10 \times m)
  \]
  where \(m\) is the number of images.

### Output Layer

- **Output Layer**: The output layer also has 10 neurons, corresponding to the digits 0-9. The output is obtained using an activation function:
  \[
  \text{output\_Layer}[1] = g(\text{hidden\_layer}[1]) = \text{ReLU}(\text{hidden\_layer}[1])
  \]

## Activation Functions

Activation functions introduce non-linearity to the model, allowing the network to capture complex patterns.

### Sigmoid Activation Function

The sigmoid function outputs a value between 0 and 1:
\[
\mathbf{s(x) = \frac{1}{1 + e^{-x}}}
\]
- **Behavior**: Values close to 1 indicate an active neuron, and values close to 0 indicate an inactive neuron. The sigmoid function pushes the input values to the ends of the curve (0 or 1), with significant changes in output for inputs around zero.

### Tanh Activation Function

The tanh function outputs a value between -1 and 1:
\[
\mathbf{tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}} = \frac{2 - (1 + e^{-2x})}{1 + e^{-2x}}= \frac{2}{1 + e^{-2x}} - 1 = 2s(2x) - 1}
\]
- **Behavior**: The tanh function is a scaled version of the sigmoid function. It pushes the input values to -1 and 1. The gradient of the tanh function is four times greater than that of the sigmoid, resulting in stronger gradients and faster learning.

### Differences Between Sigmoid and Tanh

- **Gradient**: Tanh has a higher gradient than sigmoid, leading to larger weight updates.
- **Output Symmetry**: Tanh's output is symmetric around zero, which can lead to faster convergence during training.

## Illustrations



### Sigmoid Activation Function

![Sigmoid Function]data:image/png;base64,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
)

### Tanh Activation Function

![Tanh Function](images/tanh.png)

### Forward Propagation Illustration

![Forward Propagation](images/forward_propagation.png)

## Conclusion


---
