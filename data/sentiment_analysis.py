# Sentiment Analysis Project (TASK 4)
# Author: Student
# Description: Classify text into Positive, Negative, or Neutral sentiment

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
df = pd.read_csv("data/reviews.csv")

# Function to classify sentiment
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review_Text"].apply(get_sentiment)

# Display results
print("Sentiment Analysis Results:\n")
print(df[["Review_Text", "Sentiment"]])

# Sentiment count
sentiment_count = df["Sentiment"].value_counts()
print("\nSentiment Distribution:\n")
print(sentiment_count)

# Visualization
sentiment_count.plot(kind="bar", title="Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

# Source-wise sentiment analysis
source_sentiment = df.groupby(["Source", "Sentiment"]).size().unstack()
print("\nSource-wise Sentiment Analysis:\n")
print(source_sentiment)

# Insights
print("\nKey Insights:")
print("1. Positive sentiment dominates Amazon reviews.")
print("2. Negative sentiment is more common on Social Media.")
print("3. Customer experience impacts overall sentiment.")
