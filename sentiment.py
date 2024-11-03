import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from test_data import test_data

nltk.download('vader_lexicon')

st.title("Instant Sentiment Analysis with Metrics")
text = st.text_area("Enter the text to be analyzed:")

button = st.button("Analyze!")

# Get sentiment score
def get_score(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    rounded_score = round(score['compound'], 2)
    return rounded_score

# Get predicted sentiment label based on score
def get_sentiment_label(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

# Display sentiment message
def display_message():
    if text:
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        time.sleep(1)
        my_bar.empty()

        sentiment_score = get_score(text)
        sentiment_label = get_sentiment_label(sentiment_score)

        if sentiment_label == "positive":
            sentiment = f"This is +{sentiment_score} <span style='color:lightgreen'>Positive!</span>"
        elif sentiment_label == "negative":
            sentiment = f"This is {sentiment_score} <span style='color:rgb(240, 40, 40)'>Negative!</span>"
        else:
            sentiment = "This is 0 <span style='color:grey'>Neutral!</span>"
        
        st.markdown(sentiment, unsafe_allow_html=True)

# Display sentiment score bar
def display_bar():
    sentiment_score = get_score(text)
    line_position = (sentiment_score + 1) * 50
    
    if not text:
        st.write("Score Range Indicator")
        
    score_range_html = f"""
    <div style="display: flex; align-items: center; flex-direction: column; width: 240px;">
        <div style="display: flex; width: 100%; height: 20px; border-radius: 5px; overflow: hidden; position: relative;">
            <div style="width: 25%; background-color: #ff4b4b;"></div>
            <div style="width: 25%; background-color: #ff6b6b;"></div>
            <div style="width: 25%; background-color: #6bde6b;"></div>
            <div style="width: 25%; background-color: #4bb84b;"></div>
            <div style="position: absolute; top: 0; left: calc({line_position}% - 1px); height: 20px; width: 2px; background-color: black;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; width: 100%; font-size: 0.9em; margin-top: 5px;">
            <span style="width: 25%; text-align: center; color: red; font-weight: bold;">-1</span>
            <span style="width: 25%; text-align: center; color: red;">-0.5</span>
            <span style="width: 25%; text-align: center; color: green;">+0.5</span>
            <span style="width: 25%; text-align: center; color: green; font-weight: bold;">+1</span>
        </div>
    </div>
    """
    st.markdown(score_range_html, unsafe_allow_html=True)

# Evaluate and display metrics
def print_accuracy():
    true_labels = [item["label"] for item in test_data]
    predicted_labels = [get_sentiment_label(get_score(item["text"])) for item in test_data]

    acc = accuracy_score(true_labels, predicted_labels)

    print(f"Accuracy: {acc:.2f}")

display_message()
display_bar()
print_accuracy()
