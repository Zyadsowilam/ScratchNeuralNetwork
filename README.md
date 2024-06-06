# MNIST Dataset

The MNIST database contains handwritten digits, commonly used for training various image processing systems.

## Dataset Overview

Each image in the MNIST dataset is 28x28 pixels, where the pixel values range from 0 to 255, representing black and white. The images are flattened to 784 (28x28) pixels to be used as input to a neural network.

## Neural Network Architecture

### Input Layer

- **Input Layer**: Each row represents one image. By taking the transpose of the image matrix, each column represents one image. The input layer size is 784, corresponding to the 784 pixels of each image.

### Hidden Layer

- **Hidden Layer**: The hidden layer has 10 neurons. The input to the hidden layer is calculated as follows:
  
  ```math
  \text{hidden\_layer}[1](10 \times m) = \text{Weight}[1](10 \times 784) \times \text{input\_layer}[0] + \text{bias}[1](10 \times m)
  ```
  where \(m\) is the number of images.

### Output Layer

- **Output Layer**: The output layer also has 10 neurons, corresponding to the digits 0-9. The output is obtained using an activation function:
  ```math
  \text{output\_Layer}[1] = g(\text{hidden\_layer}[1]) = \text{ReLU}(\text{hidden\_layer}[1])
  ```

## Activation Functions

Activation functions introduce non-linearity to the model, allowing the network to capture complex patterns.

### Sigmoid Activation Function

The sigmoid function outputs a value between 0 and 1:
```math
\mathbf{s(x) = \frac{1}{1 + e^{-x}}}
```
- **Behavior**: Values close to 1 indicate an active neuron, and values close to 0 indicate an inactive neuron. The sigmoid function pushes the input values to the ends of the curve (0 or 1), with significant changes in output for inputs around zero.

### Tanh Activation Function

The tanh function outputs a value between -1 and 1:
```math
\mathbf{tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}} = \frac{2 - (1 + e^{-2x})}{1 + e^{-2x}}= \frac{2}{1 + e^{-2x}} - 1 = 2s(2x) - 1}
```
- **Behavior**: The tanh function is a scaled version of the sigmoid function. It pushes the input values to -1 and 1. The gradient of the tanh function is four times greater than that of the sigmoid, resulting in stronger gradients and faster learning.

### Differences Between Sigmoid and Tanh

- **Gradient**: Tanh has a higher gradient than sigmoid, leading to larger weight updates.
- **Output Symmetry**: Tanh's output is symmetric around zero, which can lead to faster convergence during training.

## Illustrations

### Example Images from MNIST Dataset

![MNIST Examples](images/mnist_examples.png)

### Sigmoid Activation Function
![image](https://github.com/Zyadsowilam/ScratchNeuralNetwork/assets/96208685/e14b6b86-f3f5-4208-9fd6-df59896f32eb)

![Sigmoid Function](images/sigmoid.png)

### Tanh Activation Function

![Tanh Function](images/tanh.png)

### Forward Propagation Illustration

![Forward Propagation](images/forward_propagation.png)

## Conclusion

Understanding the MNIST dataset and the structure of the neural network, including the choice of activation functions, is crucial for building effective models for digit recognition. The sigmoid and tanh functions, with their distinct properties, offer different advantages in terms of gradient behavior and output symmetry.

---

This README provides a high-level overview of the MNIST dataset and the neural network architecture used for digit recognition, along with detailed explanations of key activation functions.
