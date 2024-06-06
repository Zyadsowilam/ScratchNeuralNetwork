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




## Conclusion


---


