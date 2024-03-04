"""
Visualise Sentiment Analysis Module

This module contains functions to visualise the results of the sentiment
analysis. The functions include:

- A DataFrame to show the total number of labelled review ratings and
    polarity scores: positive, negative, neutral.
- A plot to show the distribution of unlabelled polarity scores vs
    unlabelled review ratings to visualise the distributions of the
    sentiment analysis vs output labels.
- A plot to show the distribution of polarity scores vs labelled review
    ratings to visualise mis-classifications of the sentiment analysis
    model via a stacked histogram.
- A DataFrame to show the accuracy and balanced accuracy of the
    sentiment analysis.
- A confusion matrix to visualise the precision performance of the
    sentiment analysis model.
- A classification report to measure the performance of the sentiment
    analysis model.

The module is intended to be used in a Streamlit app to visualise the
results of the sentiment analysis.
"""

# Import the required libraries
from utils import pd, plt, sns, np, st
from sklearn import metrics

# Set the Streamlit page config to wide
st.set_page_config(layout="wide")

# Disable Streamlit deprecation warnings
st.set_option('deprecation.showPyplotGlobalUse', False)


def results_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create a DataFrame to show the total number of labelled 
    review ratings and polarity scores: positive, negative, neutral

    Args:
    df : pandas.DataFrame
        Input DataFrame

    Returns:
    pandas.DataFrame
        Results DataFrame
    """

    # Group the total number of ratings into positive/negative/neutral
    rating_counts = df.rating_label.value_counts()

    # Group the total number of labelled polarity scores as above
    polarity_counts = df.polarity_label.value_counts()

    # Create results DF
    result_df = pd.DataFrame({
        "Ratings Labels": rating_counts,
        "Rating Labels %": round(100 * (rating_counts / df.shape[0]), 2),
        "Polarity Labels": polarity_counts,
        "Polarity Labels %": round(100 * (polarity_counts / df.shape[0]), 2)
    })

    # Return the results DataFrame
    return result_df


def plot_polarity_distribution(df: pd.DataFrame) -> plt.figure:
    """
    Plots the distribution of unlabelled polarity scores vs
    unlabelled review ratings to visualise the distributions
    of the sentiment analysis vs output labels.

    Overlays two histograms showing polarity scores
    and review ratings.

    Args:
    df : pandas.DataFrame
        The input DataFrame

    Returns:
    fig : matplotlib.figure.Figure
        The generated figure object
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    try:

        if 'rating' not in df.columns or 'converted_polarity' not in df.columns:
            st.error("The required columns are not present in the DataFrame")
            return None

        if df.empty:
            st.error("The DataFrame is empty. There is no data to plot")
            return None

        sns.histplot(data=df, x="rating", color='red', label='Review Ratings',
                     alpha=0.3, kde=False, kde_kws={"bw_adjust": 0.5},
                     common_norm=True)
        sns.histplot(data=df, x="converted_polarity", color='blue',
                     label='Polarity Scores', alpha=0.7, kde=True,
                     kde_kws={"bw_adjust": 0.5})

        plt.title(f"Distribution of Polarity Scores: {df.name.iloc[0][:15]}")
        plt.xlabel("Reviews ratings / Polarity Scores")
        plt.ylabel("Number of Reviews")
        plt.legend()
       #plt.show()

    except KeyError as e:
        st.error(f"Column not found: {e}. \
                Please ensure correct column names in the DataFrame."
                 )
        return None

    except (TypeError, ValueError) as e:
        st.error(f"Plotting error: {e}. \
                Please ensure columns are numeric."
                 )
        return None

    # Return the figure object
    return fig


def plot_distribution_polarity_labels(df: pd.DataFrame) -> plt.figure:
    """
    Plots the distribution of polarity scores vs labelled review ratings
    to visualise mis-classifications of the sentiment analysis model
    via a stacked histogram.

    Args:
    df : pandas.DataFrame
        The input DataFrame

    Returns:
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelrotation=45)

    try:
        if ('polarity' not in df.columns) or (
                'rating_label' not in df.columns):
            st.error(
                "The required columns are not present in the DataFrame"
            )
            return None

        if df.empty:
            st.error(
                "The DataFrame is empty. There is no data to plot"
            )
            return None

        sns.histplot(data=df, x="polarity",
                     hue="rating_label", multiple="stack", kde=False)
        plt.xlim(-1, 1)
        plt.title(f"Polarity Scores vs Rating Labels: {df.name.iloc[0][:15]}"
        )
        plt.ylabel("Number of reviews")
        plt.xlabel("Polarity Scores")
        #plt.xticks(np.arange(-1, 1.1, 0.1))
        #plt.show()

    except KeyError as e:
        st.error(f"""Column not found: {e}.
                 Please ensure correct column names in the DataFrame.
                 """)
        return None

    except (TypeError, ValueError) as e:
        st.error(f"Plotting error: {e}. Please ensure columns are numeric.")
        return None

    # Return the figure object
    return fig


def accuracy_and_balanced_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the accuracy and balanced accuracy of the sentiment
    analysis.

    Args:
    df : pandas.DataFrame. 
        - Input DataFrame

    Returns:
    pd.DataFrame
        - DataFrame listing the accuracy and balanced accuracy scores
          for the sentiment analysis.

    """
    try:

        if ('rating_label' not in df.columns) or (
                'polarity_label' not in df.columns):
            st.error("The required columns are not present in the DataFrame")
            return None

        if df.empty:
            st.error("The DataFrame is empty. There is no data to plot")
            return None

    # Calculate the accuracy and balanced accuracy scores
        accuracy = 100 * round(
            metrics.accuracy_score(
                df.rating_label, df.polarity_label), 4)
        balanced_accuracy = 100 * round(
            metrics.balanced_accuracy_score(
                df.rating_label, df.polarity_label), 4)

    # Create a DataFrame with the scores
        accuracy_scores_df = pd.DataFrame(
            {'Accuracy %': [accuracy], 'Balanced Accuracy %': [
                balanced_accuracy]})

        accuracy_scores_df.set_index('Accuracy %', inplace=True)

    except KeyError as e:
        st.error(f"Column not found: {e}. \
                  Please ensure correct column names in the DataFrame."
                 )
        return None

    except (TypeError, ValueError) as e:
        st.error(f"Plotting error: {e}. Please ensure columns are numeric.")
        return None

    # Return the accuracy scores DataFrame
    print(accuracy_scores_df.columns)
    return accuracy_scores_df


def confusion_matrix(actual: pd.Series,
                     predicted: pd.Series,
                     classes: list[str] = ['Negative', 'Neutral', 'Positive']):
    """
    Plot the confusion matrix to visualise the precision performance of
    the sentiment analysis model.

    Args:
       'actual' (list): 
        - List of output labels (labelled review ratings).
        'predicted' (list): 
        - List of predicted labels (labelled polarity scores).
        classes (list, optional): 
        - List of sentiment classes. Defaults to ['Negative', 
         'Neutral', 'Positive'].

    Returns:
        matplotlib.figure.Figure: a figure object containing the
        confusion matrix plot.
    """
    try:
        # Check if the length of the actual and predicted lists are equal
        if len(actual) != len(predicted):
            st.error("""
                    Length mismatch. Ensure 'actual' and 'predicted'
                    lists have the same number of elements.
                    """
                     )
            return None
        
    
        # Create a confusion matrix
        cm = metrics.confusion_matrix(actual, predicted, normalize='pred')

        # Create a figure and plot for the confusion matrix
        fig, ax = plt.subplots()

        # Create a confusion matrix
        cm = metrics.confusion_matrix(
            actual, predicted, normalize='pred')

        # Create a confusion matrix display object
        disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes)

        # Plot the confusion matrix
        disp.plot(ax=ax)

        # Set the title of the plot
        plt.title("Confusion Matrix - Normalised (Predictions)")

        # Show the plot
        #plt.show()

    except ValueError as e:
        st.error(f"Error generating confusion matrix: {e}")
        return None

    # return the figure object
    return fig


def classification_report(actual, predicted: pd.Series) -> pd.DataFrame:
    """
    Create a classification report to measure the performance of the
    sentiment analysis model. 

    Normalise the report on predicted labels to visualise precision.
    """
    try:

        if len(actual) != len(predicted):
            st.error("""Length mismatch. Ensure 'actual' and 'predicted'
                      lists have the same number of elements.""")
            return None

        classification_report = metrics.classification_report(
            actual, predicted, output_dict=True)
        classification_df = pd.DataFrame(classification_report)

    except ValueError as e:
        st.error(f"Error generating classification report: {e}")
        return None

    return classification_df

# Descriptions of the plots for the Streamlit app

sentiment_analysis_description = """
This description provides an overview of the preprocessing steps and 
sentiment analysis performed on the CSV file.

**Preprocessing Steps** 
- Load the data
- Select relevent columns
- Drop rows with NAN values.
- Convert reviews text to lower() and strip().
- Remove stop words, punctuation, and numerical characters. 
- Return preprocessed reviews as two new columns: 
    - 1. pre-prpocessed reviews text. 
    - 2. pre-processed lemmatised reviews text.

**Sentiment Analysis** 

Use SpaCy Textblob to process token.text and token.lemma_ to return:
- Sentiment assessments.
- Polarity Score.

**Label Results** 
- Label polarity scores: "'Negative', 'Neutral', 'Positive'.
- Label review ratings: 'Negative', 'Neutral', 'Positive'.

**Interpolate Polarity Scores** 
- Perform linear transformation on polarity scores to compare to review
  ratings.

**Expected CSV columns** 

Following preprocessing and sentiment analysis carried out by the 
script sentiment_analysis.py the CSV file should contain the following
columns:
- 'name': The name of the device.
- 'reviews_text': The raw reviews text.
- 'filtered_text': The reviews text after filtering.
- 'converted_polarity_score': The converted polarity score.
- 'polarity_label': The labelled polarity score.
- 'rating': The review rating.
- 'rating_label': The labelled review rating.

"""

polarity_vs_ratings_description = """
This graph shows the distribution of the sentiment analyses predicted
polarity scores vs actual customer review ratings. A linear
transformation has been performed on the polarity scale (-1 to 1)
converting to a 1-5 scale for comparison.
"""

summary_results_description = """
This results dataframe shows the total number of: 
- Labelled customer review ratings (output labels). 
- Labelled predicted polarity scores (predicted labels).
"""

accuracy_and_balanced_accuracy_description = """
This dataframe shows the accuracy and balanced accuracy scores for the
sentiment analysis. 
"""

polarity_scores_vs_rating_labels_description = """
This stacked histplot shows the distribution of predicted polarity
scores against output rating labels (hue), visualising
the distribution of incorrect sentiment analysis polarity 
predictions.
"""

cm_description = """
This confusion matrix shows the normalised predictions of the sentiment
analysis. 'The diagonal elements represent the number of points for
which the predicted label is equal to the true label, while off-diagonal
elements are those that are misclassified by the model. The higher the
diagonal values of the confusion matrix the better, indicating many
correct predictions'."""

classification_report_description = """This classification report shows
the precision, recall, f1, and macro average scores for the sentiment
analysis. 'The precision is the ratio tp / (tp + fp) where tp is the
number of true positives and fp the number of false positives. The
recall is the ratio tp / (tp + fn) where tp is the number of true
positives and fn the number of false negatives. The F1 score is the
harmonic mean of the precision and recall, where an F1 score reaches
its best value at 1 and worst score at 0. The support is the number of
occurrences of each class in actual review ratings'."""


# Main function
def results(df: pd.DataFrame):
    """
   Execute the module functions to visualise the sentiment analysis
    results for a Streamlit app.
    """

      # Remove rows with NaN polarity and ratings labels
    df = df[(df.polarity_label != "Unknown") & (
        df.rating_label != "Unknown")].copy()

    
    # Set the title of the Streamlit app
    st.title("SpaCy TextBlob Sentiment Analysis Report")

    # Load the data from the CSV file
    st.subheader("Preprocessed & Sentiment Analysed Data")

     # Reorder the columns for improved readability
    selected_columns = ['name', 'reviews_text', 'rating',
               'rating_label', 'filtered_text', 'sentiment_assessments',
               'polarity', 'polarity_label']
    
    try:
        st.dataframe(df[selected_columns])

    except FileNotFoundError:
        # If the default file is not found, provide the file uploader
        st.subheader("Default file not found.")
        df = st.file_uploader("Upload CSV Data")

    # Description of the preprocessing and sentiment analysis steps
    st.subheader('Preproccessing and Sentiment Analysis Steps')

    try:
        st.markdown(sentiment_analysis_description)
    except Exception as e:
        st.error(f"Error occurred while displaying the description: {e}")

    # Display plots (using st.pyplot for Matplotlib plots)
    st.subheader("Polarity Scores VS Reviews Ratings")

    try:
        st.markdown(polarity_vs_ratings_description)
    except Exception as e:
        st.error(f"Error occurred while displaying the description: {e}")

    fig = plot_polarity_distribution(df)
    st.pyplot(fig)

    # Display sentiment analysis labelled results
    st.subheader("Summary Results")
    try:
        st.markdown(summary_results_description)
    except Exception as e:
        st.error(f"Error occurred while displaying the description: {e}")

    st.dataframe(results_df(df))

    # Display plot for distribution of polarity labels
    st.subheader("Polarity Scores vs Ratings Labels")
    try:
        st.markdown(polarity_scores_vs_rating_labels_description)
    except Exception as e:
        st.error(f"Error occurred while displaying the description: {e}")
    fig = plot_distribution_polarity_labels(df)
    st.pyplot(fig)

    # Return the accuracy and balanced accuracy scores df
    st.subheader("Accuracy Score")
    try:
        st.markdown(accuracy_and_balanced_accuracy_description)
    except Exception as e:
        st.error(f"Error occurred while displaying the description: {e}")

    st.dataframe(accuracy_and_balanced_accuracy(df))

    # Plot the confusion matrix normalised to predictions
    st.subheader("Confusion Matrix 'Normalised Predictions'")
    try:
        st.markdown(cm_description)
    except Exception as e:
        st.error(f"Error occurred while displaying the description: {e}")

    fig = confusion_matrix(df.rating_label, df.polarity_label,
                           classes=['Negative', 'Neutral', 'Positive'])
    st.pyplot(fig)

    # Display the classification report
    st.subheader('Classification Report')
    try:
        st.markdown(classification_report_description)
    except Exception as e:
        st.error(f"Error occurred while displaying the description: {e}")
    st.write(classification_report(df.rating_label,
                                   df.polarity_label))
    
    
    return None