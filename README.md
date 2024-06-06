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
![image](https://github.com/Zyadsowilam/ScratchNeuralNetwork/assets/96208685/e14b6b86-f3f5-4208-9fd6-df59896f32eb)
- **Behavior**: Values close to 1 indicate an active neuron, and values close to 0 indicate an inactive neuron. The sigmoid function pushes the input values to the ends of the curve (0 or 1), with significant changes in output for inputs around zero.

### Tanh Activation Function

The tanh function outputs a value between -1 and 1:
```math
\mathbf{tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}} = \frac{2 - (1 + e^{-2x})}{1 + e^{-2x}}= \frac{2}{1 + e^{-2x}} - 1 = 2s(2x) - 1}
```
![image](https://github.com/Zyadsowilam/ScratchNeuralNetwork/assets/96208685/81853179-0203-4259-9f9a-c4b78ff17700)

- **Behavior**: The tanh function is a scaled version of the sigmoid function. It pushes the input values to -1 and 1. The gradient of the tanh function is four times greater than that of the sigmoid, resulting in stronger gradients and faster learning.

### Differences Between Sigmoid and Tanh

- **Gradient**: Tanh has a higher gradient than sigmoid, leading to larger weight updates.
- **Output Symmetry**: Tanh's output is symmetric around zero, which can lead to faster convergence during training.
![image](https://github.com/Zyadsowilam/ScratchNeuralNetwork/assets/96208685/ca306eeb-e854-425c-8d73-33ab66518bac)
Rectified Linear Activation Function

The sigmoid and hyperbolic tangent activation functions cannot be used in networks with many layers due to the vanishing gradient problem.
The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better.
The ReLU (Rectified Linear Unit) activation function is defined as follows:

```math
\text{ReLU}(x) = \begin{cases} 
0 & \text{if } x \leq 0 \\
x & \text{if } x > 0 
\end{cases}
```

In mathematical notation, it can be expressed as:

```math
\text{ReLU}(x) = \max(0, x)
```

This function outputs the input value \( x \) if \( x \) is greater than 0, and 0 otherwise. It's a simple yet effective activation function widely used in neural networks, especially in deep learning models.

![image](https://github.com/Zyadsowilam/ScratchNeuralNetwork/assets/96208685/431e8d4a-36e3-43ea-b289-843a1868981c)

### Softmax
The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or negative, the softmax turns it into a small probability, and if an input is large, then it turns it into a large probability, but it will always remain between 0 and 1.The softmax function is a commonly used activation function, especially in the output layer of a neural network for multi-class classification problems. It converts raw scores or logits into probabilities that sum up to 1. The softmax function is defined as follows:

Given a vector \( \mathbf{z} = (z_1, z_2, ..., z_n) \) of raw scores (logits), the softmax function \( \text{softmax}(z_i) \) for each element \( z_i \) is calculated as:

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
```

This function exponentiates each element of the input vector \( \mathbf{z} \), then divides each exponentiated value by the sum of all exponentiated values in the vector, ensuring that the resulting values form a probability distribution that sums to 1. 

In vectorized form, the softmax function for a vector \( \mathbf{z} \) can be written as:

```math
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
```

![image](https://github.com/Zyadsowilam/ScratchNeuralNetwork/assets/96208685/f644c663-98d1-4f93-b0a0-4e76ebad9668)

### Backpropagation
Backpropagation, short for "backward propagation of errors," is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error function, the method calculates the gradient of the error function with respect to the neural network's weights. It is a generalization of the delta rule for perceptrons to multilayer feedforward neural networks.


## Conclusion


---


