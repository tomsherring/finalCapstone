"""
Perform Sentiment Analysis Module

This module contains functions to perform sentiment analysis on
preprocessed text using SpaCy TextBlob. The module also contains
functions to label the polarity scores and review ratings, and to
convert the polarity scores to a 1-5 rating scale for comparison
with the review ratings in a separate visualisation script.

The sentiment analysis is performed using SpaCy TextBlob, which
provides a polarity score and sentiment assessments for the text.
The polarity score is a float value between -1 and 1, where -1 is
most negative and 1 is most positive. The sentiment assessments
are a list of tuples containing the words and their sentiment
scores.
"""

# Import libraries
from utils import pd, spacy, np, st
from spacytextblob.spacytextblob import SpacyTextBlob

def sentiment_analysis(text: str) -> pd.DataFrame:
    """
    Perform sentiment analysis on preprocessed text using SpaCy
    TextBlob.

    Parameters:
    - text (str): the customer review text.

    Returns:
    - assessments (list): the sentiment assessments.
    - polarity (float): the polarity score.
    """

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")

    # Basic error handling
    try:
        if not text or not isinstance(text, str):
            return pd.Series(["", None],  # Empty assessments, None for polarity
                             index=['sentiment_assessments', 'polarity'])

        doc = nlp(text)

        # Extract sentiment assessments and polarity score
        assessments = doc._.blob.sentiment_assessments.assessments
        polarity = doc._.blob.polarity

        return pd.Series({
            'sentiment_assessments': [", ".join(str(x) for x in assessments)],
            'polarity': polarity,
        })

    except TypeError as e:
        print(f"Input error: Text must be a string. Error: {e}")
        return pd.Series(["", None], index=['sentiment_assessments', 'polarity'])

    except AttributeError as e:
        print(f"Error accessing SpaCy attributes: {e}")
        return pd.Series(["", None], index=['sentiment_assessments', 'polarity'])

    except ValueError as e:
        print(f"Text processing error: {e}")
        return pd.Series(["", None], index=['sentiment_assessments', 'polarity'])

def label_polarity_scores(polarity: float) -> str:
    """
    Assign labels to the SpaCy TextBlob polarity scores: positive,
    negative, or neutral.

    Uses a simple if-else statement to map the polarity scores to the
    labels.

    Parameters:
    - polarity (float): the polarity score.

    Returns:
    - label (str): the label assigned to the polarity score:
    Positive, Negative, Neutral, (Unknown in case of NaN).
    """
    if pd.isna(polarity):
        return "Unknown"
    elif polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"

def label_review_ratings(rating: int) -> str:
    """
    Assign label to the review rating: positive, negative,
    neutral using a dictionary mapping.

    Parameters:
    - rating (int8): the review rating.

    Returns:
    - label (str): the label assigned to the review rating:
    Positive, Negative, Neutral, (Unknown in case of NaN).
    """
    rating_labels = {
        1: "Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Positive",
    }
    return "unknown" if pd.isna(rating) else rating_labels[rating]


def convert_polarity_to_rating(polarity: float) -> float:
    """
    Converts the polarity scores from a -1 to 1 to 1-5 rating scale,
    using Numpy.interp() to achieve a linear transformation. This
    will allow for comparison with the review ratings on two overlaid 
    histograms in the separate visualisation script.

    Parameters:
    - polarity (float): the polarity score.

    Returns:
    - rating (float): the converted rating score.
    """
    polarity_min = -1
    polarity_max = 1
    rating_min = 1
    rating_max = 5

    try:
        # Return NaN if input is NaN
        if pd.isna(polarity):
            return np.nan

    # Error handling for potential exceptions
    except Exception as e:
        print(f"Error in convert_polarity_to_rating {polarity} Error: {e}")

    # Return the converted rating score
    return np.interp(polarity, (polarity_min, polarity_max), (rating_min, rating_max))


def process_reviews(df) -> pd.DataFrame:
    """
    Calls the above functions reviews to perform sentiment
    analysis on preprocessed text.

    Returns:
    - df (Pandas DataFrame): df containing the processed data and
    sentiment analysis results.
    """

    try:

        # Perform sentiment analysis on preprocessed text to
        df[['sentiment_assessments', 'polarity']] = df['reviews_text'].apply(
            sentiment_analysis
        )
    except (ValueError, KeyError) as e:
        print(f"Error during sentiment analysis: {e}")

    try:

        # Label polarity scores; set as categorical d-type to save memory
        df["polarity_label"] = (
            df["polarity"].apply(
                label_polarity_scores).astype("category"))

    except (ValueError, KeyError) as e:
        print(f"Error during polarity labeling: {e}")

    try:
        # Label review ratings. Set to category data type to save memory
        df["rating_label"] = (
            df["rating"].apply(label_review_ratings).astype("category")
        )
    except (ValueError, KeyError) as e:
        print(f"Error during rating labeling: {e}")
    
    try:
        # Convert the polarity scores to a 1-5 rating scale for comparison
        df["converted_polarity"] = df["polarity"].apply(
            convert_polarity_to_rating
        )

    except (ValueError, KeyError) as e:
        print(f"Error during polarity conversion: {e}")

    return df
