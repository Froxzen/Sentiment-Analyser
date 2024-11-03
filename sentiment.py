import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('vader_lexicon')
st.title("Instant Sentiment Analysis")
text = st.text_area("Enter the text to be analyzed:")
button = st.button("Analyze!")

# Test data for accuracy prediction
test_data = [
    {"text": "I love this product!", "label": "positive"},
    {"text": "This is the worst service ever.", "label": "negative"},
    {"text": "I'm feeling great today!", "label": "positive"},
    {"text": "I don't like this.", "label": "negative"},
    {"text": "It's okay, not bad.", "label": "neutral"}
]

def score():
    sia = SentimentIntensityAnalyzer()                                          
    score = sia.polarity_scores(text)
    rounded_score = round(score['compound'], 2)
    return rounded_score

def display_message():
    if text:
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        time.sleep(1)
        my_bar.empty()

        if score() > 0:
            sentiment = f"This is +{score()} <span style='color:lightgreen'>Positive!</span>"
        elif score() < 0:
            sentiment = f"This is {score()} <span style='color:rgb(240, 40, 40)'>Negative!</span>"
        else:
            sentiment = "This is 0 <span style='color:grey'>Neutral!</span>"
        st.markdown(sentiment, unsafe_allow_html=True)


def display_bar():
    # Calculate line position based on compound score
    line_position = (score() + 1) * 50 
    
    if not text:
        st.write("Score Range Indicator")
        
    # HTML and CSS for the score range bar with a line indicating the score 
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
    # Render the score range bar with the line
    st.markdown(score_range_html, unsafe_allow_html=True)

display_message()
display_bar()
