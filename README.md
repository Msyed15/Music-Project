# Music-Project
Music Emotion Recognition Project
Overview
This project aims to classify songs by their emotional tone using machine learning and Natural Language Processing (NLP) techniques. The dataset consists of song lyrics and audio features, and the goal is to build a model that predicts the emotion of a song based on both its lyrics and audio attributes.

Objectives:
Data Preprocessing: Clean and structure the song data for efficient analysis.

Feature Extraction: Derive meaningful features from the lyrics (e.g., sentiment analysis, TF-IDF) and audio characteristics.

Modeling: Build a machine learning model to classify emotions based on the extracted features.

Visualization: Generate visualizations to display key insights and model performance.

Advanced Techniques: Implement advanced NLP techniques such as sentiment analysis and ensemble learning methods like Random Forest and Gradient Boosting to improve model performance.

Part I: Data Preprocessing
Data Cleaning
The first step involved cleaning the dataset, which included handling missing values, ensuring consistent formatting for text data, and normalizing the audio features to standardize the input data for model training.

Code for Data Cleaning:

<img width="597" alt="Screenshot 2025-05-02 at 9 52 10 PM" src="https://github.com/user-attachments/assets/2c58f186-7c38-4ec3-addb-7095bf2ff4d1" />


Extracting Text Features
Next, the notebook extracts features from song lyrics using a TF-IDF vectorizer to capture the most important words that influence emotional predictions:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)  # Was 5000
X_tfidf = tfidf.fit_transform(combined_df['clean_lyrics'].fillna("")).toarray()


Visualizing Feature Importance
The notebook visualizes the top features that are most important for predicting the song's emotional tone using a lollipop chart:

<img width="799" alt="Screenshot 2025-05-02 at 10 14 34 PM" src="https://github.com/user-attachments/assets/33adc248-b319-4ec2-8c2a-072eea0f4dc2" />


<img width="989" alt="Screenshot 2025-05-02 at 9 41 04 PM" src="https://github.com/user-attachments/assets/8cdabe5e-3ac3-444c-9199-978f85689c0f" />

Part II: Model Building
Preparing the Data for Training
The labels are re-encoded, and a Logistic Regression model is prepared for training:

<img width="948" alt="Screenshot 2025-05-02 at 10 16 00 PM" src="https://github.com/user-attachments/assets/8381c869-ae09-4737-aed0-459aa55ffa13" />


<img width="790" alt="Screenshot 2025-05-02 at 9 45 44 PM" src="https://github.com/user-attachments/assets/6c7dfb20-e875-4a3e-8fa7-3ae9a36c0461" />


Part III: Model Evaluation
The model's accuracy and classification results are evaluated using a classification report, providing a detailed view of how well the model performs on different emotional categories.

<img width="779" alt="Screenshot 2025-05-02 at 9 46 07 PM" src="https://github.com/user-attachments/assets/b05d0588-9f07-4cb4-8fd4-27bb9acb88c1" />

Confusion Matrix
We also use a confusion matrix to visualize how many instances of each emotion were correctly or incorrectly predicted. This is an important diagnostic tool to identify areas where the model might be confused between categories.

Code for Confusion Matrix:

<img width="622" alt="Screenshot 2025-05-02 at 10 18 28 PM" src="https://github.com/user-attachments/assets/409c2780-92e7-4adc-b55f-1958dfef63a5" />


<img width="583" alt="Screenshot 2025-05-02 at 9 46 44 PM" src="https://github.com/user-attachments/assets/c4aaf730-fa49-4254-be8b-de67ca2860a7" />

Future Work and Improvements
Hyperparameter Tuning
To further improve model performance, hyperparameter tuning can be explored. For instance, adjusting the number of estimators in the Random Forest or experimenting with different ensemble models could yield better results.

Deep Learning Models
We can experiment with deep learning models like LSTM (Long Short-Term Memory) or Transformers to better capture the sequential nature of lyrics, potentially improving the emotion prediction accuracy.

Larger Dataset
Increasing the diversity of the dataset to include songs from various genres and languages could help generalize the model and make it more robust for real-world applications.











