# Exp 7 Stock Market Prediction using Linear Regression and Real-Time Sentiment Analysis of Tweets

**Date:**

## AIM:

To implement **Stock Market Prediction using Linear Regression** to predict future stock prices using machine learning regression techniques and **Real-Time Sentiment Analysis of Tweets** to analyze user-provided text data such as tweets or reviews and classify their sentiment.

## DESIGN STEPS:

### Step 1:

Clone the repository from GitHub.

### Step 2:

Create a Python project in the preferred IDE (VS Code/PyCharm/Jupyter Notebook).

### Step 3:

Create the Python program for implementing Stock Market Prediction using Linear Regression and Real-Time Sentiment Analysis using suitable Python libraries.

### Step 4:

Load the historical stock market dataset and select the required features such as **Open, High, Low, Volume**, and **Close** price.

### Step 5:

Preprocess the stock market dataset and split it into training and testing data.

### Step 6:

Train the **Linear Regression** model using the training data and predict the stock prices for the testing data.

### Step 7:

Evaluate the stock prediction model using suitable regression metrics and visualize the actual and predicted stock prices.

### Step 8:

Accept user-provided text data such as **tweets, reviews, or comments** as input for sentiment analysis.

### Step 9:

Preprocess the input text by converting it into lowercase and removing unnecessary characters and punctuation.

### Step 10:

Calculate the **sentiment polarity score** of the input text using Natural Language Processing techniques.

### Step 11:

Classify the input text as **Positive, Negative, or Neutral** based on the polarity score.

### Step 12:

Execute the program and analyze the stock price prediction and sentiment analysis results.

## PROGRAM:
#STOCK_DATA
```
from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("stock_data_big.csv")

print(data.head())

X = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nMean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

plt.figure(figsize=(8,5))
plt.plot(y_test.values[:50], label='Actual Prices', color='blue')
plt.plot(predictions[:50], label='Predicted Prices', color='red')
plt.title("Stock Price Prediction (Actual vs Predicted)")
plt.xlabel("Time")
plt.ylabel("Stock Close Price")
plt.legend()
plt.show()
```
#TWEETS_DATA
```
uploaded = files.upload()
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

data = pd.read_csv("tweets_big.csv")

def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"


data["Sentiment"] = data["text"].apply(get_sentiment)

sentiment_counts = data["Sentiment"].value_counts()
print(sentiment_counts)

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green','red','gray'])
plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Tweets/Reviews")
plt.show()

print("\nSample Results:")
print(data[["text", "Sentiment"]].head())
```
## OUTPUT:
<img width="305" height="103" alt="Screenshot 2026-08-25 112107" src="https://github.com/user-attachments/assets/4ef2156c-9b9f-45b4-8afc-85ed07be32e2" />
<img width="305" height="103" alt="Screenshot 2026-08-25 112107" src="https://github.com/user-attachments/assets/c14bb065-2cc8-413f-831e-aada8cb85ef0" />
<img width="305" height="103" alt="Screenshot 2026-08-25 112107" src="https://github.com/user-attachments/assets/f762d341-06bf-49b2-9da2-5fb3e21b2909" />
<img width="305" height="103" alt="Screenshot 2026-08-25 112107" src="https://github.com/user-attachments/assets/d1d446d5-6a8e-4bd7-b773-13dbd4f42f24" />
<img width="305" height="103" alt="Screenshot 2026-08-25 112107" src="https://github.com/user-attachments/assets/8c286022-acd4-480f-aee6-273dc50767cd" />


## RESULT:

The **Stock Market Prediction using Linear Regression** and **Real-Time Sentiment Analysis of Tweets** were implemented successfully. The Linear Regression model was used to predict stock prices using historical stock market data, while the sentiment analysis system successfully analyzed user-provided tweets or reviews and classified them as **Positive, Negative, or Neutral**.
