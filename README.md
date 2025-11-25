# Object Detection Model using PyTorch

## Project Overview

This project implements an object detection model using PyTorch to identify cow stall numbers and their bounding boxes from images. The model combines classification and bounding box regression tasks to achieve multi-task learning on the cow stall dataset.

## Learning Objectives

Through this project, I learned:

- **PyTorch Deep Learning Framework**: Building custom neural network architectures using `nn.Module`
- **Transfer Learning**: Leveraging pre-trained ResNet50 model for improved performance
- **Computer Vision Fundamentals**: 
  - Image preprocessing and augmentation using torchvision transforms
  - Bounding box detection and regression
  - Handling image data with OpenCV and PIL
- **Multi-task Learning**: Simultaneous classification and bounding box regression
- **Model Training & Optimization**:
  - Loss function design for multi-task scenarios
  - Learning rate scheduling with StepLR
  - Training loop implementation with validation
  - Performance metrics tracking (accuracy, loss)
- **Data Handling**: Custom PyTorch Dataset and DataLoader implementation

## Dataset

**Dataset Used**: [Cow Stall Number Dataset](https://github.com/YoushanZhang/Cow_stall_number)

The dataset contains:
- Training and test splits with image paths and annotations
- Labels: Cow stall numbers (61 classes)
- Bounding box coordinates (x, y, width, height)
- CSV format annotations for easy loading

## Model Architecture

The `ObjectDetection` class combines:
- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Feature Extraction**: 2048-dimensional feature vector
- **Classification Head**: Linear layer (2048 → 61 classes)
- **Bounding Box Head**: Linear layer (2048 → 4 coordinates)

```python
class ObjectDetection(nn.Module):
    - ResNet50 backbone for feature extraction
    - Classification branch for stall number prediction
    - Bounding box regression branch for localization
```

## Key Components

### 1. **Data Preprocessing**
- Image resizing to 224×224 pixels
- Normalization using ImageNet statistics
- Data augmentation (horizontal flip, vertical flip, rotation)
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 2. **Custom Dataset Class** (`CowstallDataset`)
- Reads images from disk using OpenCV
- Extracts labels and bounding boxes from CSV
- Applies transformations during data loading
- Returns image, label, and bounding box coordinates

### 3. **Training Process**
- **Optimizer**: Adam with learning rate = 0.0001 and weight decay = 0.0001
- **Loss Functions**:
  - Classification Loss: Cross-Entropy Loss
  - Bounding Box Loss: MSE Loss (scaled by 0.01)
  - Combined Loss: loss = loss1 + √(loss2)
- **Learning Rate Schedule**: StepLR (step_size=30, gamma=0.1)
- **Epochs**: 100

### 4. **Evaluation Metrics**
- **Accuracy**: Classification accuracy for stall number prediction
- **Loss**: Combined loss for both tasks

## Results

✅ **Model Performance**: **85.06% Validation Accuracy**

**Final Metrics:**
- Train Loss: 0.0626
- Train Accuracy: 100.00%
- Validation Loss: 0.7947
- Validation Accuracy: 85.06%

The model successfully exceeded the 80% accuracy threshold requirement.

## Training Visualizations

The notebook includes loss visualization showing:
- Training loss progression across epochs
- Convergence behavior
- Model learning effectiveness

## Technical Stack

- **Deep Learning Framework**: PyTorch
- **Computer Vision**: torchvision, OpenCV (cv2)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Development Environment**: Google Colab (with GPU support)

## File Structure

```
Project_2_Jainam_.ipynb
├── 1. Build an object detection model using PyTorch
│   ├── Library imports
│   ├── Device setup (CUDA/CPU)
│   ├── Data transforms
│   ├── Model architecture (ObjectDetection)
│   └── Custom dataset class (CowstallDataset)
│
├── 2. Train model using Cow Stall Number Dataset
│   ├── Data loading
│   ├── Loss function definition
│   ├── Optimizer setup
│   ├── Training loop (100 epochs)
│   └── Model checkpoint saving
│
├── 3. Results & Visualization
│   ├── Training loss plot
│   ├── Performance metrics (85.06% accuracy)
│   └── Research paper link
```

## How to Use

### Prerequisites
```
torch
torchvision
opencv-python
pandas
numpy
matplotlib
scikit-learn
```

### Running the Notebook

1. **Setup**: Mount Google Drive and load the dataset
2. **Data Loading**: Load training and test CSV files
3. **Model Training**: Execute training cells (100 epochs)
4. **Evaluation**: Check final accuracy and loss metrics
5. **Visualization**: View training loss curves

## Key Learnings & Insights

1. **Multi-task Learning Power**: Combining classification and bounding box regression improved overall performance
2. **Transfer Learning Effectiveness**: Pre-trained ResNet50 reduced training time and improved accuracy
3. **Loss Function Design**: Balancing multiple loss terms (with weighted factor 0.01) is crucial for convergence
4. **Learning Rate Scheduling**: StepLR helped prevent overfitting and improve generalization
5. **Data Augmentation**: Random transformations improved model robustness
6. **Batch Size Impact**: Batch size of 4 provided stable training with available resources

## Model Weights & Research Paper

The trained model weights and LaTeX research paper are available at:
[Google Drive Link](https://drive.google.com/drive/folders/1cN8CLvk38omU1SRo13yMKsDRXc3k0ZOD?usp=sharing)

The research paper contains:
- Detailed methodology
- Experimental setup
- Results analysis
- Comparison with baselines

## Future Improvements

- Implement anchor-based detection (YOLO/Faster R-CNN)
- Add non-maximum suppression for better box predictions
- Experiment with different backbone architectures
- Implement IoU (Intersection over Union) metric for box evaluation
- Deploy model for real-time inference

## References

- [PyTorch Documentation](https://pytorch.org/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Cow Stall Number Dataset](https://github.com/YoushanZhang/Cow_stall_number)
- ResNet: He et al., "Deep Residual Learning for Image Recognition"

## Author

Jainam Bhansali

## Date

Project completed: 2024

---

**Note**: This project was part of a neural network course assignment focusing on practical implementation of object detection using modern deep learning frameworks.
