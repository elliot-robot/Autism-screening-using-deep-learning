# Autism-screening-using-deep-learning
ResNet50-based framework integrates multimodal data (videos, audio, questionnaires) for early ASD diagnosis. Achieves high accuracy with explainable AI, addressing subjective traditional methods. Enhances accessibility in underserved regions, improving intervention outcomes and life quality.

📁 GitHub Repository: Autism Screening Using Deep Learning (ResNet50)
👤 Author: Saikiran Challa | 🎓 Degree: Master of Data Science | 🏫 University: La Trobe University

📌 Overview
This repository contains the code and resources for a deep learning framework designed to screen Autism Spectrum Disorder (ASD) using facial images of children. The model leverages ResNet50 with transfer learning and achieves 88.5% accuracy and 95.5% AUC, outperforming benchmarks like EfficientNetB0. The project addresses challenges in traditional ASD diagnosis by providing an automated, scalable, and objective screening tool.

✨ Key Features
Model Architecture: ResNet50 with residual connections to mitigate vanishing gradients.

Data Augmentation: Techniques like rotation, flipping, and zooming to enhance generalization.

Optimization: Adamax optimizer with a learning rate of 0.001 for superior convergence.

Explainability: Integration of regularization (dropout, L2) to reduce overfitting.

Benchmarking: Comparative analysis with EfficientNetB0 and other architectures.

📊 Results
Metric	ResNet50	EfficientNetB0
Accuracy	88.50%	87.00%
AUC	95.50%	94.70%
Visualizations:

Training vs. Test Accuracy/Loss curves.

ROC-AUC and Precision-Recall plots.

🛠️ Installation & Usage
Prerequisites
Python 3.8+

TensorFlow 2.12+

NumPy, Matplotlib, Scikit-learn


```bash
# Directory Structure
autism-screening-resnet50/
├── data/                    # Preprocessed datasets
│   ├── train/              # Training images
│   ├── test/               # Testing images
│   └── val/                # Validation images
├── models/                 # Saved model weights
│   └── resnet50_best.h5
├── src/
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── utils.py           # Helper functions
├── notebooks/             # EDA and testing notebooks
├── requirements.txt       # Dependencies
└── README.md
```

🧩 Key Contributions
Algorithm Design: Custom ResNet50 architecture for ASD-specific feature extraction.

Clinical Relevance: Integration with real-world workflows and interpretability for clinicians.

Scalability: Tools for deployment in underserved regions with limited diagnostic resources.

```bash
# Clone the repository
git clone https://github.com/yourusername/autism-screening-resnet50.git
cd autism-screening-resnet50

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py --model resnet50 --epochs 20 --batch_size 64

# Evaluate performance
python src/evaluate.py --model_path models/resnet50_best.h5
```

```python
# Sample Training Code Snippet (src/train.py)
import tensorflow as tf
from utils import load_data, build_resnet50

# Load dataset
train_data, val_data = load_data('data/train', 'data/val')

# Build model
model = build_resnet50(input_shape=(224, 224, 3), num_classes=2)

# Train
model.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
    batch_size=64,
    callbacks=[tf.keras.callbacks.ModelCheckpoint('models/resnet50_best.h5')]
)
```

🤝 Contributing
Contributions are welcome! Open an issue or submit a PR for:

Multi-modal data integration (audio, video, questionnaires).

Real-time deployment pipelines.

Enhanced explainability using Grad-CAM/Saliency maps.

📄 License
This project is licensed under the MIT License.

📧 Contact
For questions or collaborations:

Email: 21356161@students.ltu.edu.au

LinkedIn: Saikiran Challa

🌟 Star this repo if you find it useful!
🔗 Dataset Link: [Kaggle ASD Dataset](https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set)

