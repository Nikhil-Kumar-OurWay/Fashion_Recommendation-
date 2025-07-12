*Fashion Product Recommendation System*

This project is a content-based image recommendation system built using deep learning (ResNet50) and k-NN, allowing users to upload a fashion product image and receive similar product recommendations.

🧵 *Project Description*
This project is a Fashion Recommendation Model that leverages deep learning and computer vision techniques to suggest visually similar fashion products based on a user's input image. The system allows users to upload a fashion product photo—like a shirt, dress, shoe, or accessory—and instantly receive visually related recommendations from a preloaded image dataset.

At its core, the model uses a pre-trained ResNet50 convolutional neural network (CNN) to extract deep visual features from images. These features are then compared using the k-Nearest Neighbors (k-NN) algorithm, which identifies and returns the top 5 most visually similar items from the dataset.

The entire backend pipeline, including model development and feature extraction, was implemented in Google Colab. The extracted features and image paths were saved as .pkl files (Images_features.pkl and filenames.pkl), which are later used in a Streamlit web application. This app provides a user-friendly interface for uploading images and viewing recommendations in real-time.

🧰 Libraries Used
✅ Core Libraries:
os – File and path handling
numpy – Numerical and array operations
pickle – Saving and loading serialized model/data

✅ Deep Learning & Image Processing:

tensorflow – Deep learning framework (used for loading ResNet50)
keras.applications.resnet50 – Pre-trained ResNet50 model and preprocessing
keras.preprocessing.image – Loading and converting input images
keras.layers.GlobalMaxPool2D – Feature flattening layer after CNN

✅ Machine Learning:
sklearn.neighbors.NearestNeighbors – To find similar images using k-NN

✅ Web Application:
streamlit – For building the interactive user interface

✅ Optional/Notebook-only:

matplotlib.pyplot – For displaying images (used in Jupyter notebook)
cv2 (OpenCV) – For image processing and display (if used in notebook)

📂 Project Structure

├── app.py                          
├── Fashion_Recommendation_Model.ipynb  
├── Images_features.pkl            
├── filenames.pkl                  
├── upload/ 

⚙️ Tech Stack

1) Python 🐍
2) TensorFlow / Keras
3) ResNet50 (pre-trained CNN)
4) Scikit-learn (NearestNeighbors)
5) Streamlit (for the web interface)
6) NumPy, Pickle