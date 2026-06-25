# Lightweight MNIST Classification System based on TinyML

## Project Introduction
Aiming at the limitations of computing power, memory and power consumption of edge embedded devices, this project designs a lightweight CNN image classification model based on TensorFlow.
Through INT8 post-training quantization technology, the model is compressed to obtain an edge-side AI inference model deployable on microcontrollers, which verifies the feasibility of TinyML technology on low-resource devices.

## Tech Stack
- Deep Learning Framework: TensorFlow 2.10
- Edge Inference Engine: TensorFlow Lite
- Core Technology: Lightweight Convolutional Neural Network, INT8 Weight Quantization

## Performance Results
| Model Type | Accuracy | Model Size | Inference Time per Sample |
| :--------- | :------- | :--------- | :------------------------ |
| Original Float Model | 98.63% | 788.53 KB | - |
| TinyML Quantized Model | 98.10% | 66.84 KB | 1.155 ms |

- Volume compression ratio: 11.8x
- Accuracy loss: 0.53 percentage points

## File Description
- `tinyml_mnist.py`: Complete project code, including dataset loading, model construction, training, quantization and inference verification. The results can be reproduced by running the script directly.
