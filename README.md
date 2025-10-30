# DeepFake Video Detection

This project named as DeepFake Video Detection,used to detect any video is deepfake or not, is a web app whose backend host a pre trained model  and runs in flask. You can run this project on your local machine as well as on cloud very easily.


### Overview
This project focuses on developing machine learning models for video processing with applications in detecting fake videos. The core technology leverages deep learning architectures such as GRU (Gated Recurrent Units) for sequence modeling and InceptionV3 and ResNet50 for feature extraction from video frames. The project is designed to preprocess videos, train models to recognize patterns indicative of video authenticity, and evaluate model performance, highlighting key metrics such as training and validation accuracy.

### Features
- Video Preprocessing: Standardize video frames to a fixed size and format to feed into the neural network.
- Feature Extraction: Utilize InceptionV3 and ResNet architecture to extract meaningful features from each video frame.
- Sequence Modeling: Implement GRU layers to capture temporal dependencies between consecutive video frames.
  
### Model Training and Validation:
- Train models on a labeled dataset with metrics displayed each epoch for monitoring.
- Detailed tracking of training and validation accuracy to assess performance and generalizability.
- Use callbacks like ModelCheckpoint and EarlyStopping to optimize training and prevent overfitting.
- Model Optimization: Adjust learning rates and model architecture based on performance metrics.
  
### Model Performance
the model demonstrated consistent improvement in both training and validation loss, indicating effective learning and adaptation to the training data. Here is a summary of the model's performance over the training period:
- Training Accuracy: Started at 68.05% and improved to 80.60% by the end of the training.
- Validation Accuracy: Remained consistent at 81.25%, suggesting that the model generalizes well to new data.

This performance trajectory indicates that the model is stable and effectively learns the distinguishing features between classes without overfitting, as evidenced by the parallel improvement in validation metrics.

## Dependencies
- TensorFlow
- Keras
- NumPy
- OpenCV
- Pandas

### References:
https://www.researchgate.net/publication/368589748_Detection_of_Deepfake_Video_Using_Residual_Neural_Network_and_Long_Short-Term_Memory




# Environment Setup

Make sure Anaconda is installed and launch anaconda prompt and navigate to root directory in the anaconda prompt

create venv

```shell
conda create -n deepfakedetection python=3.10
```

Activate

```shell
conda activate deepfakedetection 
```

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

run the app.py file 

```shell
python app.py
```

Once you see this url - http://127.0.0.1:5000/ in logs, open it in browser.


Now your setup is ready.
