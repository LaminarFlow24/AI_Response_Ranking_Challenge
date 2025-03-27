

# Response Winner Prediction
## This Project gave us **First Prize** in **"Microsoft Student Learn Chapter PCCOE Hackathon"**.  

This project compares responses from two models (model_a and model_b) based on their similarity to a prompt. Using transformer-based sentence embeddings and additional text features, the project trains a LightGBM classifier to predict which model’s response is better or if the result is a tie.

## Project Overview

- **Data Processing:**  
  Cleans input text by removing extra spaces and unwanted characters. The dataset includes prompts and responses (from two models).

- **Feature Engineering:**  
  - **Text Similarity:** Uses the `all-MiniLM-L6-v2` model from SentenceTransformer to compute embeddings. Cosine similarity between the prompt and each response is calculated.
  - **Additional Features:** Extracts word and character counts from prompts and responses, including the difference in word counts between responses.
  - **TF-IDF Features:** Combines all text columns and extracts TF-IDF features.
  - **Categorical Encoding:** Encodes model labels.

- **Model Training:**  
  A LightGBM classifier is trained using a pipeline that includes preprocessing of text, numeric, categorical, and additional features. Hyperparameter tuning is performed with GridSearchCV.

- **Evaluation & Prediction:**  
  The model is evaluated using accuracy, classification reports, and confusion matrices. Finally, predictions on the test dataset are saved into a `submission.csv` file.

## Requirements

- Python 3.x
- [pandas]
- [numpy]
- [scikit-learn]
- [lightgbm]
- [sentence-transformers]
- [tqdm]

If using Google Colab, the notebook includes steps to download the output CSV files directly.

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/response-winner-prediction.git
   cd response-winner-prediction

2. **Install Dependencies**
   pip install pandas numpy scikit-learn lightgbm sentence-transformers tqdm

3. **(Optional) Using Google Colab**
   Upload your dataset files (train.csv and test.csv) to your Google Drive.
   Update the file paths in the notebook accordingly.
   Use the provided code snippet for downloading the output CSV files.



