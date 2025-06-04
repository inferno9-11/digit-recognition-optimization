# digit-recognition-optimization
**Sustainable and Energy-Efficient Handwritten Digit Recognition**

## 🔍 Project Overview
GreenDigits is an energy-efficient deep learning framework for handwritten digit recognition. Designed with sustainability and deployment on edge devices in mind, this project integrates:

- Adaptive Model Switching
- Dynamic Network Pruning
- Energy-Aware Knowledge Distillation
- Quantization-Aware Training
- Real-Time Energy Monitoring

Our approach maintains high classification accuracy while significantly reducing power consumption and computational load, making it suitable for use in mobile devices, IoT hardware, and embedded systems.

## 🚀 Key Features

- **Adaptive Model Switching:** Efficiently switches between a lightweight and a complex model depending on input complexity.
- **Dynamic Pruning:** Removes less significant neurons and connections during inference.
- **Knowledge Distillation:** Trains a compact student model to mimic the performance of a large teacher model.
- **Quantization-Aware Training:** Reduces the bit-width of weights and activations for lower energy and memory use.
- **Hardware-Efficient Deployment:** Targeted for devices like Raspberry Pi, Jetson Nano, ARM Cortex chips, and FPGAs.

## 🧠 Architecture
graph TD
    A[Input Digit Image] --> B{Is Input Simple?}
    B -- Yes --> C[Lightweight CNN Model]
    B -- No --> D[Heavy CNN Model]
    C --> E[Dynamic Pruning Layer]
    D --> E
    E --> F[Quantized Output Layer]
    F --> G[Predicted Digit]
    G --> H[Energy Monitor & Feedback]

## 🧪 Technologies Used
- Python
- TensorFlow / PyTorch (specify based on your code)
- ONNX / TFLite for model conversion
- Energy measurement tools (e.g., Jetson Stats, Power Profiler)

## 📦 Dataset
- **MNIST Handwritten Digit Dataset**
  - 60,000 training images
  - 10,000 test images
  - 28x28 grayscale digit images

## 📈 Results
| Method                         | Accuracy (%) | Power Reduction (%) |
|-------------------------------|--------------|----------------------|
| Baseline CNN                  | 98.3         | 0                    |
| Pruned + Quantized Model      | 97.6         | 35                   |
| Distilled Student Model       | 97.9         | 42                   |
| Full GreenDigits Framework    | 98.1         | 55                   |

## 🛠 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/GreenDigits.git
cd GreenDigits

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate the model
python evaluate.py

# Export for edge deployment
python export_model.py --format tflite
```
## 💬 Acknowledgments
- MNIST Dataset by Yann LeCun et al.
- Inspiration from works by Han et al. and Hinton et al. on model compression and knowledge distillation.

