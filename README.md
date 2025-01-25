# Image Recognition Project: Cat vs Dog

This project demonstrates a deep learning model for image classification that distinguishes between images of cats and dogs. Using Convolutional Neural Networks (CNNs), it processes labeled images and predicts whether the input image is a cat or a dog.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This project uses TensorFlow/Keras to build and train a CNN for binary image classification. The model achieves robust performance in distinguishing cats from dogs, making it a good starting point for basic image recognition tasks.

---

## Dataset
The dataset consists of images of cats and dogs, divided into training and testing sets:
- **Training Set**: Used for training the CNN.
- **Testing Set**: Used to evaluate model performance.

### Directory Structure
```
ImageRecognitionProject-Cat-vs-Dog-
├── dataset
│   ├── training_set
│   │   ├── cats
│   │   └── dogs
│   ├── test_set
│       ├── cats
│       └── dogs
├── model
│   └── saved_model.h5
├── main.py
└── README.md
```

---

## Model Architecture
The CNN model uses the following layers:
1. **Convolutional Layer**: Extracts features from input images.
2. **MaxPooling Layer**: Reduces spatial dimensions of feature maps.
3. **Fully Connected Layer**: Combines extracted features to classify images.
4. **Activation Function**: ReLU for intermediate layers and Sigmoid for the output layer.

Key Features:
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy

---

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Pranavsingh431/ImageRecognitionProject-Cat-vs-Dog-.git
   cd ImageRecognitionProject-Cat-vs-Dog-
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset is placed in the `dataset` folder as shown above.

---

## Usage

### Training the Model
Run the following command to train the model:
```bash
python main.py
```

### Testing the Model
Modify `main.py` to load the saved model and test it on custom images or the test set.

### Predicting a Single Image
Add your image to the dataset folder and modify the code to predict using:
```python
model.predict(image)
```

---

## Results
- Achieved an accuracy of approximately **XX%** on the test set.
- Model visualization and performance metrics can be found in the output logs.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to your fork:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).
