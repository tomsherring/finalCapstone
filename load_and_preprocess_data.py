
"""
Load And Preprocess Data Module

This module contains functions to load and preprocess the customer
reviews data. The module also contains a function to load and preprocess
the data in a single function call.

The data is loaded from a CSV file and cleaned to remove any rows with
empty values in the target columns. The most reviewed Amazon device is
selected and a sample of the data is returned to avoid memory issues.

The customer reviews text data is preprocessed using SpaCy to:
- convert to lower case.
- strip white space.
- remove stop words.
- remove punctuation.
- remove numeric characters.
- lemmatise the text.

The preprocessed text is returned as a new column in the DataFrame.
"""

# Import libraries
import os
from utils import pd, np, spacy, st

# Remove 'no' and 'not' from SpaCy's 'stop word' dictionary
cls = spacy.util.get_lang_class("en")
cls.Defaults.stop_words.remove("no")
cls.Defaults.stop_words.remove("not")

# Use OS module create a filepath to the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create a relative file path to the CSV file
file_path = os.path.join(script_dir, "amazon_product_reviews.csv")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the data from the CSV file and perform initial data cleaning.
    Select the most reviewed Amazon device and return a sample of the
    data.

    Parameters:
    - file_path (str): relative file path to the CSV file

    Returns:
    - df (Pandas DataFrame): loaded and cleaned data
    """
    try:
        # Import data using Pandas
        # Set low_memory false to avoid warning about mixed data types
        df = pd.read_csv(file_path, low_memory=False)

    # Error handling for common exceptions loading data
    except FileNotFoundError as e:
        print(f"File not found: {e}. Check file path and name")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"Empty file: {e}. Check file and formatting")
        return None
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}. Check data and formatting")
        return None

    # Select columns to read in from the CSV file
    selected_columns = ["name", "reviews.text", "reviews.rating"]

    # Drop any rows with empty values in the target columns
    df = df.dropna(subset=["name", "reviews.text",
                   "reviews.rating"]).reset_index(drop=False)

    # Select top Amazon device by number of reviews
    most_reviewed_device = df["name"].value_counts().nlargest(1).index

    # Create a df copy featuring only the most reviewed device
    df = df[df["name"].isin(most_reviewed_device)].copy()

    # Specify data types for the columns to avoid mixed data types
    # NP.int8 saves memory and is sufficient for the rating scale
    dtypes = {"name": str, "reviews.text": str, "reviews.rating": np.int8}

    # Cast data to correct data types
    df = df[selected_columns].astype(dtypes, errors="ignore")

    # Rename columns for improved readability
    df.rename(
        columns={"reviews.text": "reviews_text", "reviews.rating": "rating"},
        inplace=True,
    )

    sample_size = 1000
    # Return a sample of the data to avoid memory issues
    return df.sample(sample_size, random_state=42)
    # Return the cleaned DataFrame with selected columns
    # return df

def preprocess_reviews(text: str) -> pd.DataFrame:
    """
    Preprocess the customer reviews text data:

    Python:
    - convert to lower case.
    - strip white space.

    SpaCy:
    - remove stop words.
    - remove punctuation (results in a limited handling of contractions
      when paired with lemmatisation).
    - remove numeric characters.
    - lemmatise the text.

    Parameters:
    - text (str): the customer review text.

    Returns:
    - filtered_text (str): the preprocessed and lemmatised text.
    """

    # Load SpaCy SM model
    nlp = spacy.load("en_core_web_sm")

    try:
        if not text or not isinstance(text, str):
            return pd.Series(["", ""], index=['filtered_text'])

        # Force string data to lower case and strip white space
        text = text.lower().strip()

        # Create SpaCy doc object.
        doc = nlp(text)

        # Filter tokens returning token.text
        filtered_text = [
            token.lemma_ for token in doc if not token.is_stop
            and token.is_alpha
        ]

        # Return filtered text as a string
        return pd.Series({
            'filtered_text': " ".join(filtered_text)})

    # Error handling for common string processing exceptions
    except UnicodeDecodeError:
        print(f"Encoding Error: Check text data. Input Text: {text}")
        return pd.Series(["", ""], index=['filtered_text'])

    except TypeError:
        print(
            f"TypeError: Likely an empty input. Returning empty row. Input Text: {text}")
        return pd.Series(["", ""], index=['filtered_text'])

    except Exception as e:
        print(f"Unexpected error. Input Text: {text} Error: {e}")
        return pd.Series(["", ""], index=['filtered_text'])


def load_and_preprocess():
    """
    Load and preprocess the customer reviews data

    Returns:
    - df (Pandas DataFrame): loaded and preprocessed data

    """
    # Load the data
    df = load_data(file_path)
    # Preprocess the data
    df['filtered_text'] = df['reviews_text'].apply(
        preprocess_reviews
    )
    # Return the preprocessed DataFrame
    return df
    