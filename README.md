# Brain Tumor Detection System using Deep Learning


This repository contains the code and resources for the implementation of a Brain Tumor Detection System using Deep Learning. The system is based on a Sequential CNN (Convolutional Neural Network) model trained on a dataset of brain tumor images.

## Overview
- Developed a deep learning-based binary classification model to detect brain tumors.
- Used a Sequential CNN architecture with multiple convolutional and dense layers.
- Achieved high accuracy of 98.5% in tumor detection on the test dataset.
- Implemented data augmentation techniques to enhance model performance.
- Utilized the TensorFlow and Keras libraries for model development and training.
- Deployed the system using the Streamlit framework to provide a user-friendly interface.

## Project Structure
- `data/`: Contains the dataset of brain tumor images (preprocessed and augmented).
- `models/`: Includes the saved trained model and model evaluation scripts.
- `src/`: Contains the source code for data preprocessing, model training, and testing.
- `streamlit/`: Includes the Streamlit web application for the user interface.
- `requirements.txt`: Specifies the required dependencies for running the project.

## Usage
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Preprocess and augment the dataset using the scripts in the `src/` directory.
4. Train the model using the `train_model.py` script and save the trained model.
5. Evaluate the model using the `evaluate_model.py` script.
6. Launch the Streamlit web application using `streamlit run streamlit/app.py` and interact with the system.

## Results
- The model achieved an accuracy of 98.5% in detecting brain tumors.
- Confusion matrix:
    - True Positive: 100
    - False Positive: 5
    - False Negative: 3
    - True Negative: 192
- Precision: 0.952
- Recall: 0.971
- F1-Score: 0.961

## Future Work
- Improve the model's performance by exploring advanced CNN architectures.
- Enhance the user interface by adding more interactive features.
- Extend the system to detect and classify different types of brain tumors.
- Investigate the potential of transfer learning for better accuracy and efficiency.


Dataset - https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection



