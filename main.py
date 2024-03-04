 """
    Main module function to load, preprocess, perform sentiment analysis and
    visualise the customer reviews data.

    - Load and preprocess the customer reviews data using the 
      load_and_preprocess_data.py module.

    - Perform sentiment analysis using the 
      perform_sentiment_analysis.py module.

    - Visualise the sentiment analysis using the 
      visualise_sentiment_analysis.py module.

    The program can be executed to generate a sentiment analysis csv
    report, or run as a Streamlit app to visualise the sentiment
    analysis results.

    Returns:
    - df (Pandas DataFrame): df containing the processed data and
    sentiment analysis results.

    """

import load_and_preprocess_data as preprocess
import perform_sentiment_analysis as sentiment
import visualise_sentiment_analysis as visualise

def main():
   
    # Load and preprocess the data
    df = preprocess.load_and_preprocess()

    # Perform sentiment analysis
    df = sentiment.process_reviews(df)

    # Visualise the sentiment analysis
    visualise.results(df)

    df.to_csv('sentiment_analysis.csv', index=False)

if __name__ == "__main__":
    main()
