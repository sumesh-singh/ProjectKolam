# Kolam AI Recognition and Recreation System

This document describes the complete AI system for kolam (rangoli) pattern recognition and recreation, forming the core of the Kolam Design System project.

## Overview

The Kolam AI system consists of two main deep learning models:

1. **Convolutional Neural Network (CNN)** - For recognizing and classifying kolam patterns
2. **Generative Adversarial Network (GAN)** - For generating new kolam patterns

Both models are integrated into a cohesive pipeline that can analyze existing kolam patterns and create new variations while preserving the traditional aesthetic qualities.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kolam AI Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Data       │  │   CNN       │  │    GAN      │             │
│  │  Pipeline   │  │   Model     │  │   Model     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Recognition  │  │ Generation  │  │ Evaluation  │             │
│  │   Engine    │  │   Engine    │  │   System    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Files Structure

### Core Model Files

- `kolam_cnn_model.py` - CNN model for pattern recognition
- `kolam_gan_model.py` - GAN model for pattern generation
- `kolam_data_pipeline.py` - Data preprocessing and augmentation
- `kolam_ai_pipeline.py` - Integrated recognition and generation pipeline
- `kolam_model_evaluation.py` - Model evaluation and testing utilities

### Test and Demo Files

- `test_kolam_models.py` - Comprehensive test suite for all models

### Dataset

- `rangoli_dataset_complete/` - Complete dataset with images and metadata
  - `images/` - Organized by region and state
  - `metadata/` - JSON files with pattern information
  - `logs/` - Scraping and processing logs

## Installation and Setup

### Prerequisites

- Python 3.8+
- TensorFlow 2.14.0
- PyTorch 2.1.0
- OpenCV, scikit-image, numpy, pandas
- matplotlib, seaborn for visualizations

### Installation

All required dependencies are already listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Quick Start with Test Suite

Run the comprehensive test suite to verify all models:

```python
python backend/test_kolam_models.py
```

This will test:

- Data pipeline functionality
- CNN model training and evaluation
- GAN model training and generation
- Integrated AI pipeline

### 2. Individual Model Usage

#### CNN Model for Pattern Recognition

```python
from kolam_cnn_model import KolamCNNModel

# Initialize model
cnn_model = KolamCNNModel(img_height=224, img_width=224)

# Load and preprocess data
images, labels, class_names = cnn_model.load_dataset('backend/rangoli_dataset_complete/images')
X_train, X_test, y_train, y_test = cnn_model.preprocess_data(images, labels)

# Build and train
model = cnn_model.build_model()
history = cnn_model.train_model(X_train, y_train, X_test, y_test)

# Make predictions
results = cnn_model.predict('path/to/kolam_image.jpg')
print(f"Predicted class: {results['predicted_label']}")
print(f"Confidence: {results['confidence']:.2%}")
```

#### GAN Model for Pattern Generation

```python
from kolam_gan_model import KolamGAN

# Initialize GAN
gan_model = KolamGAN(img_height=128, img_width=128)

# Train on dataset
gan_model.train('backend/rangoli_dataset_complete/images', epochs=1000)

# Generate new patterns
new_patterns = gan_model.generate_kolam(num_images=5)

# Save generated images
for i, pattern in enumerate(new_patterns):
    plt.imshow(pattern)
    plt.savefig(f'generated_kolam_{i}.png')
```

### 3. Integrated Pipeline

```python
from kolam_ai_pipeline import KolamAIPipeline

# Initialize complete pipeline
pipeline = KolamAIPipeline()

# Train both models
training_results = pipeline.train_full_pipeline(
    dataset_path='backend/rangoli_dataset_complete/images',
    metadata_path='backend/rangoli_dataset_complete/metadata',
    cnn_epochs=50,
    gan_epochs=1000
)

# Use for recognition and generation
analysis = pipeline.analyze_and_generate('input_kolam.jpg', num_variations=5)
print(f"Input pattern recognized as: {analysis['input_analysis']['predicted_label']}")
print(f"Generated {analysis['num_variations']} variations")
```

## Model Details

### CNN Architecture

The CNN model uses a progressive architecture with:

- **Data Augmentation**: Random flip, rotation, zoom for better generalization
- **Convolutional Blocks**: 4 blocks with increasing filter sizes (32→256)
- **Batch Normalization**: For stable training
- **Dropout**: To prevent overfitting (0.5 and 0.3 rates)
- **Dense Layers**: 512 and 256 units with ReLU activation
- **Output**: Softmax classification layer

**Key Features:**

- Input size: 224×224×3 (configurable)
- Total parameters: ~2.1M (trainable)
- Mixed precision training support
- Early stopping and learning rate scheduling

### GAN Architecture

The GAN consists of:

#### Generator

- **Input**: 100-dimensional noise vector
- **Architecture**: Transposed convolutions with upsampling
- **Output**: 128×128×3 generated image
- **Activation**: Tanh for [-1, 1] output range

#### Discriminator

- **Input**: 128×128×3 image (real or generated)
- **Architecture**: Convolutional blocks with downsampling
- **Output**: Single sigmoid output (real/fake probability)
- **Activation**: LeakyReLU for better gradient flow

**Training:**

- Binary cross-entropy loss
- Adam optimizer with β₁=0.5
- Label smoothing for stability
- Progressive image saving during training

## Data Pipeline

### Features

- **Automatic Dataset Loading**: Recursive directory traversal
- **Metadata Integration**: JSON metadata support
- **Data Augmentation**: Comprehensive image augmentation
- **Class Balancing**: Oversampling for imbalanced datasets
- **Multiple Splits**: Train/validation/test splits
- **Normalization**: Multiple normalization methods

### Dataset Organization

The dataset is organized hierarchically:

```
images/
├── region_1/
│   ├── state_1/
│   │   ├── pattern_1.jpg
│   │   └── pattern_2.jpg
│   └── state_2/
├── region_2/
└── ...
```

### Metadata Format

Each image has corresponding metadata:

```json
{
  "filename": "pattern_1.jpg",
  "folder": "region/state",
  "source_url": "https://...",
  "search_query": "kolam type",
  "description": "Pattern description",
  "download_timestamp": "2025-09-28T...",
  "file_size": 12345,
  "source": "pinterest"
}
```

## Evaluation and Metrics

### CNN Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Confusion Matrix**: Detailed prediction analysis
- **ROC AUC**: Multi-class ROC analysis
- **Inference Time**: Model speed measurement

### GAN Evaluation Metrics

- **Diversity Score**: How different generated images are
- **Quality Score**: Similarity to real images
- **Discriminator Accuracy**: How well discriminator distinguishes real/fake
- **Stability Score**: Training stability over time

### Visualization

- Training history plots
- Confusion matrices
- Generated image galleries
- Class distribution charts
- Loss progression graphs

## Performance Optimization

### Hyperparameter Tuning

- Grid search and random search support
- Learning rate optimization
- Batch size tuning
- Architecture parameter optimization

### Training Optimizations

- Mixed precision training (FP16)
- Batch normalization
- Early stopping
- Learning rate scheduling
- Model checkpointing

### Inference Optimizations

- Model quantization support
- TensorRT optimization ready
- Batch inference support

## Deployment Preparation

### Model Serialization

- **CNN**: Saved as HDF5 format with training history
- **GAN**: Separate generator and discriminator models
- **Pipeline**: Complete pipeline state preservation

### API Integration

The models are designed for integration with:

- **FastAPI**: REST API endpoints
- **WebSocket**: Real-time generation
- **Database**: Results storage and retrieval
- **Caching**: Redis for performance

### Scalability

- **Batch Processing**: Multiple image handling
- **Cloud Deployment**: AWS/GCP ready
- **Docker**: Containerization support
- **Monitoring**: Prometheus metrics

## Testing

### Test Coverage

- Unit tests for individual components
- Integration tests for complete pipeline
- Performance benchmarks
- Accuracy validation
- Generation quality assessment

### Running Tests

```bash
# Run all tests
python backend/test_kolam_models.py

# Run specific component tests
python -c "from kolam_cnn_model import KolamCNNModel; print('CNN import successful')"
python -c "from kolam_gan_model import KolamGAN; print('GAN import successful')"
```

## Troubleshooting

### Common Issues

1. **Memory Issues**

   - Reduce batch size
   - Use smaller image sizes
   - Enable mixed precision training

2. **Training Instability (GAN)**

   - Adjust learning rates
   - Use label smoothing
   - Increase discriminator training frequency

3. **Poor CNN Accuracy**

   - Increase data augmentation
   - Try different architectures
   - Balance dataset classes
   - Use transfer learning

4. **Import Errors**
   - Check Python path
   - Verify TensorFlow/PyTorch installation
   - Ensure all dependencies are installed

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features

- **Conditional GAN**: Generate patterns based on style/type
- **Style Transfer**: Apply kolam styles to user images
- **Progressive GAN**: Higher resolution generation
- **Attention Mechanisms**: Better feature extraction
- **Ensemble Methods**: Multiple model combination

### Research Directions

- **Few-shot Learning**: Train with limited data
- **Domain Adaptation**: Handle different kolam styles
- **Interpretability**: Explain model decisions
- **Ethical AI**: Cultural sensitivity preservation

## Contributing

### Code Style

- PEP 8 compliance
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features

### Adding New Models

1. Follow the existing architecture patterns
2. Implement evaluation metrics
3. Add to test suite
4. Update documentation

## License

This project is part of the Kolam Design System and follows the project's licensing terms.

## Support

For issues and questions:

1. Check existing documentation
2. Review test outputs
3. Examine log files
4. Create detailed issue reports with:
   - Error messages
   - System configuration
   - Dataset information
   - Expected vs actual behavior

---

**Generated on**: 2025-09-29
**Version**: 1.0.0
**Status**: Complete AI Pipeline Ready for Integration
