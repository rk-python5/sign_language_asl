# Sign Language Recognition Using Deep Learning

A real-time American Sign Language (ASL) recognition system that uses computer vision and deep learning to translate hand gestures into text. The system supports 24 ASL letters (A-Y, excluding J and Z) and includes a "NOTHING" class for non-gesture frames.

## Features

- **Real-time Recognition**: Live webcam-based sign language recognition with smooth prediction display
- **Multi-Model Architecture**: Implements both CNN (Keras/TensorFlow) and Transformer-based (PyTorch) models
- **Advanced Hand Detection**: Uses MediaPipe for robust hand landmark extraction
- **Multiple Preprocessing Techniques**: Various image enhancement methods for better accuracy
- **Comprehensive Testing**: Batch testing capabilities with confidence scoring
- **Model Visualization**: Training curves and accuracy metrics visualization

## Technologies Used

### Core Technologies
- **Python 3.8+**
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand landmark detection and tracking
- **PyTorch** - Deep learning framework for transformer model
- **TensorFlow/Keras** - CNN model implementation
- **NumPy** - Numerical computations

### Additional Libraries
- **Matplotlib** - Visualization and plotting
- **Pandas** - Data manipulation
- **Scikit-learn** - Data preprocessing utilities

## Model Architecture

### 1. CNN Model (Keras/TensorFlow)
The CNN model processes 28x28 grayscale images with the following architecture:

```
- Conv2D (128 filters, 5x5 kernel) + MaxPool2D
- Conv2D (64 filters, 2x2 kernel) + MaxPool2D  
- Conv2D (32 filters, 2x2 kernel) + MaxPool2D
- Flatten
- Dense (512 units) + Dropout (0.25)
- Dense (24 units, softmax) - Output layer
```

### 2. Transformer Model (PyTorch)
Advanced transformer-based architecture using hand landmarks:

```
- Input Projection (3D landmarks → 256D)
- Multi-Head Attention (4 heads)
- Feed-Forward Networks with Dropout
- Layer Normalization
- Global Average Pooling
- Classification Head (24 classes)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/SignLanguageRecognition.git
   cd SignLanguageRecognition
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Additional PyTorch dependencies:**
   ```bash
   pip install torch torchvision torchaudio
   ```

## Dataset

The system works with ASL letter datasets containing:
- **Training Data**: Images organized in folders (A-Y, excluding J and Z)
- **Test Data**: Individual test images for validation
- **Format**: JPG/PNG images with hand gestures

### Dataset Structure
```
dataset/
├── train_set/
│   ├── A/
│   ├── B/
│   └── ...
└── test_set/
    ├── A_test.jpg
    ├── B_test.jpg
    └── ...
```

## How to Train

### 1. CNN Model Training
```bash
cd CNN
python model.py
```

### 2. Transformer Model Training
```bash
cd scripts
python train_model.py
```

### Training Parameters
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 64
- **Learning Rate**: 0.0005
- **Optimizer**: AdamW with weight decay
- **Validation Split**: 80/20

## Real-Time Prediction

Run the real-time recognition system:

```bash
cd scripts
python real_time_test.py
```

### Controls
- **Press 'q'** to quit the application
- **Webcam**: Uses default camera (index 0)
- **Resolution**: 1280x720 for optimal performance

## Preprocessing Steps

The system employs multiple preprocessing techniques:

1. **Image Enhancement**:
   - Bilateral filtering for noise reduction
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Gaussian blur for smoothing

2. **Hand Detection**:
   - Multiple MediaPipe configurations
   - Different confidence thresholds
   - Various image preprocessing variations

3. **Data Augmentation**:
   - Random brightness/contrast adjustment
   - Small rotation angles (-10° to +10°)
   - Enhanced preprocessing pipeline

## Testing

### Batch Testing
```bash
cd scripts
python test_model.py
```

### Model Evaluation
```bash
cd scripts
python evaluate_model.py
```

## Evaluation

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Confidence Scores**: Prediction confidence levels
- **Training Curves**: Loss and accuracy visualization
- **Per-Class Performance**: Individual letter recognition rates

### Expected Performance
- **Training Accuracy**: >95%
- **Validation Accuracy**: >90%
- **Real-time Performance**: 10+ FPS

## Project Structure

```
Sign-Language-Recognition/
├── CNN/                          # CNN model implementation
│   ├── model.py                  # Keras CNN model
│   └── cnn_data.zip             # CNN training data
├── scripts/                      # Main application scripts
│   ├── train_model.py           # PyTorch model training
│   ├── real_time_test.py        # Real-time recognition
│   ├── test_model.py            # Batch testing
│   ├── evaluate_model.py        # Model evaluation
│   ├── dataconvertor.py         # Data preprocessing
│   └── load_files.py            # Utility functions
├── dataset/                      # Training and test data
│   ├── train_set/               # Training images
│   ├── test_set/                # Test images
│   └── landmarks_dataset.csv    # Processed landmark data
├── models/                       # Saved model files
│   ├── sign_language_model.pth  # Trained PyTorch model
│   └── landmarks_dataset.npz    # Processed dataset
├── images/                       # Visualization outputs
│   ├── model_accuracy.png       # Training accuracy curves
│   ├── model_loss.png           # Training loss curves
│   └── training_curves.png      # Combined training metrics
├── docs/                         # Documentation
│   └── README.pdf               # Detailed documentation
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Supported Gestures

The system recognizes 24 ASL letters:

**A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y**

**Note**: Letters J and Z are excluded as they require motion for proper recognition.

## Future Work

1. **Motion Recognition**: Add support for dynamic gestures (J, Z)
2. **Multi-Hand Support**: Recognition of two-handed signs
3. **Sentence Recognition**: Complete word and phrase recognition
4. **Mobile Deployment**: iOS/Android app development
5. **Improved Accuracy**: Advanced data augmentation and model architectures
6. **Real-time Translation**: Text-to-speech integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MediaPipe** team for the excellent hand tracking solution
- **PyTorch** and **TensorFlow** communities for deep learning frameworks
- **OpenCV** contributors for computer vision tools

## Contact

**Author**: Rehaan Khatri
**Email**: rehaankh7@gmail.com
**GitHub**: [@rk-python5](https://github.com/rk-python5)

---

For detailed technical documentation, please refer to the [README.pdf](docs/README.pdf) file in the docs directory.