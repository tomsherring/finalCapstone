# Hyperion Dev - Data Science (Fundamentals) Boot Camp - Final Capstone Project: SpaCy TextBlob Sentiment Analysis

## Overview

This project demonstrates sentiment analysis of customer reviews using SpaCy and TextBlob. It includes modules for data preprocessing, sentiment analysis, and visualization, providing insights into customer opinions expressed in product reviews.

### Features

#### Data Preprocessing:
Cleans and prepares customer review text data.
Removes stop words, punctuation, and numeric characters.
Lemmatizes text for linguistic analysis.

#### Sentiment Analysis:
Utilizes SpaCy TextBlob to extract sentiment assessments.
Calculates polarity scores (positive/negative sentiment).
Labels polarity scores and review ratings for interpretability.

#### Visualizations (Streamlit App):
Distribution of polarity scores vs. review ratings.
Polarity scores vs. labeled review ratings (for analyzing model performance).
Accuracy and balanced accuracy scores.
Confusion matrix for precision visualization.
Classification report for detailed performance metrics.

### Dataset

#### Source: 
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
#### Description: 
The dataset is Datafiniti's 'Consumer Reviews of Amazon Products,' publically availabile via Kaggle. The dataset lists 'over 34,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick, and more' derived from 'Datafiniti's Product Database. The dataset includes basic product information, rating, review text, and more for each product.' The dataset needs to be downloaded, then saved as 'amazon_product_reviews.csv' in the projects root folder.

### Technologies

Python,
Streamlit,
SpaCy,
TextBlob,
pandas,
NumPy,
scikit-learn,
Matplotlib, 
Seaborn,

### Installation

#### 1. Clone Repository: 
git clone https://github.com/tomsherring/finalCapstone
#### 2. Create Virtual Environment: 
python -m venv env
source env/bin/activate

#### 3. Install Dependencies: 
pip install -r requirements.txt

### Usage

#### Prepare CSV:
Ensure you have downloaded and renamed the the Datafiniti Kaggle Dataset as 'amazon_product_reviews.csv,' saving to the repository's root folder.

#### Run main.py:
python3 main.py

Generates a CSV containing the results of the data preprocessing and sentiment analysis

#### Run Streamlit app: 
streamlit run main.py

Processes the input data and returns visualiations of the results, indludinng Seaborn histplots and Sci-kit Learn evaluation metrics.

#### Required CSV Columns

name: Product name
reviews.text: Raw review text

reviews.rating: Customer rating (1-5 scale)

##### Other columns generated by the scripts:
filtered_text
sentiment_assessments
polarity
polarity_label
rating
rating_label

### Contributing

This project welcomes contributions and suggestions. Feel free to open issues or submit pull requests to enhance its functionality!

### Project Structure

#### load_and_preprocess_data.py:
Module for loading and preprocessing review data.
#### perform_sentiment_analysis.py: 
Module for performing sentiment analysis using SpaCy TextBlob.
#### visualise_sentiment_analysis.py: 
Module for creating visualizations of the sentiment analysis results.
#### main.py:
The main script coordinating data loading, analysis, and visualization

### Example Screenshots

<img width="1424" alt="Screenshot 2024-03-04 at 16 22 39" src="https://github.com/tomsherring/finalCapstone/assets/64788094/f4ad307b-be9e-476f-8066-9f426b8915fd">
![image](https://github.com/tomsherring/finalCapstone/assets/64788094/22fbea36-52e0-4cf4-ac75-6a6ea571bc4d)
![image](https://github.com/tomsherring/finalCapstone/assets/64788094/bfa23cb0-2a1a-4f1b-9b87-8e5079aba606)
![image](https://github.com/tomsherring/finalCapstone/assets/64788094/6a94212e-aa00-4a2a-9fc1-e8972b49ae6d)










