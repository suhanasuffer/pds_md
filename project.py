import requests
import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from youtube_comment_downloader import YoutubeCommentDownloader
from itertools import islice
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, \
    classification_report, roc_curve, auc
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from wordcloud import WordCloud
from sklearn.preprocessing import label_binarize

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

def create_pie_chart(positive_percentage, negative_percentage, neutral_percentage):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive_percentage, negative_percentage, neutral_percentage]
    colors = ['lightgreen', 'lightcoral', 'lightskyblue']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')  

    return fig

# Using VADER function to label sentiment of data after comment extraction
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Getting comments using YoutubeCommentDownloader() from youtube_comment_downloader library
def get_comments(video_url, num_comments=1000):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url)

    comment_list = list(islice(comments, num_comments))
    comment_texts = [comment['text'] for comment in comment_list]

    sentiment_labels = [analyze_sentiment(text) for text in comment_texts]
    comments_df = pd.DataFrame({'Comment': comment_texts, 'Sentiment': sentiment_labels})

    return comments_df

# creating function for word cloud
def create_wordcloud(text_data):
    combined_text = ' '.join(text_data)

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(combined_text)

    # Display the word cloud using Streamlit
    st.image(wordcloud.to_array())

# function to get titles of the youtube videos using beautifulsoup library
def get_video_title(video_url):
    try:
        response = requests.get(video_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title_element = soup.find("meta", property="og:title")
            if title_element:
                return title_element["content"]
    except Exception as e:
        st.write("Error getting video title:", e)

# function to calculate gini index
def calculate_gini_index(sentiments):
    positive_count = (sentiments == 'Positive').sum()
    negative_count = (sentiments == 'Negative').sum()
    total_count = len(sentiments)
    p1 = (positive_count / total_count) ** 2
    p2 = (negative_count / total_count) ** 2
    gini_index = 1 - (p1 + p2)
    return gini_index

# function to plot curve
def plot_roc_curve(y_test, predicted_probabilities, positive_class_label):
    y_test_binary = label_binarize(y_test, classes=[positive_class_label, 'Others'])
    fpr, tpr, _ = roc_curve(y_test_binary[:, 0], predicted_probabilities[:, 0])

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Positive vs. Others)')
    plt.legend(loc="lower right")
    st.pyplot(plt)

def LR(video_url, num_comments=1000):
    comments_df = get_comments(video_url, num_comments)

    if comments_df.empty or len(comments_df) < 2:
        st.write("Error: Not enough comments to perform analysis.")
        return None, None, None, None, None, None, None, None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(comments_df['Comment'], comments_df['Sentiment'],
                                                        test_size=0.3, random_state=42)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    predicted_sentiments_lr = lr_model.predict(X_test_tfidf)

    # Calculate metrics for Logistic Regression
    accuracy_LR = accuracy_score(y_test, predicted_sentiments_lr)
    precision_LR = precision_score(y_test, predicted_sentiments_lr, average='weighted')
    recall_LR = recall_score(y_test, predicted_sentiments_lr, average='weighted')
    f1_LR = f1_score(y_test, predicted_sentiments_lr, average='weighted')
    jaccard_LR = jaccard_score(y_test, predicted_sentiments_lr, average='weighted')
    gini_positive_lr = calculate_gini_index(y_test[predicted_sentiments_lr == 'Positive'])
    gini_negative_lr = calculate_gini_index(y_test[predicted_sentiments_lr == 'Negative'])

    # auc-roc
    predicted_probabilities_lr = lr_model.predict_proba(X_test_tfidf)
    st.write("ROC curve for LR")
    plot_roc_curve(y_test, predicted_probabilities_lr, 'Positive')
    # true positive
    true_positive_rate = recall_score(y_test, predicted_sentiments_lr, labels=['Positive'], average=None)
    # false positive
    false_positive_rate = 1 - recall_score(y_test, predicted_sentiments_lr, labels=['Negative'], average=None)
    # confusion matrix
    confusion = confusion_matrix(y_test, predicted_sentiments_lr)
    # classification report
    report = classification_report(y_test, predicted_sentiments_lr, target_names=['Negative', 'Neutral', 'Positive'])
    # sensitivity analysis
    sensitivity_covariate = st.sidebar.number_input("Change in Covariate-LR(e.g., 0.1 for 10% change):", 0.0, 1.0, 0.1)
    sensitivity_covariate = str(sensitivity_covariate)
    sensitivity_prediction = lr_model.predict(tfidf_vectorizer.transform([sensitivity_covariate]))

    return comments_df, accuracy_LR, precision_LR, recall_LR, f1_LR, jaccard_LR, gini_positive_lr, gini_negative_lr, true_positive_rate, false_positive_rate, confusion, report, sensitivity_covariate, sensitivity_prediction

# Function to compare models for K-Nearest Neighbors
def KNN(video_url, num_comments=1000):
    comments_df = get_comments(video_url, num_comments)

    if comments_df.empty or len(comments_df) < 2:
        st.write("Error: Not enough comments to perform analysis.")
        return None, None, None, None, None, None, None, None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(comments_df['Comment'], comments_df['Sentiment'],
                                                        test_size=0.3, random_state=42)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
    knn_model.fit(X_train_tfidf, y_train)
    predicted_sentiments_knn = knn_model.predict(X_test_tfidf)

    # Calculate metrics for K-Nearest Neighbors
    accuracy_knn = accuracy_score(y_test, predicted_sentiments_knn)
    precision_knn = precision_score(y_test, predicted_sentiments_knn, average='weighted')
    recall_knn = recall_score(y_test, predicted_sentiments_knn, average='weighted')
    f1_knn = f1_score(y_test, predicted_sentiments_knn, average='weighted')
    jaccard_knn = jaccard_score(y_test, predicted_sentiments_knn, average='weighted')
    gini_positive_knn = calculate_gini_index(y_test[predicted_sentiments_knn == 'Positive'])
    gini_negative_knn = calculate_gini_index(y_test[predicted_sentiments_knn == 'Negative'])

    # auc-roc
    predicted_probabilities_knn = knn_model.predict_proba(X_test_tfidf)
    st.write("ROC curve for KNN")
    plot_roc_curve(y_test, predicted_probabilities_knn, 'Positive')
    # true positive
    true_positive_rate = recall_score(y_test, predicted_sentiments_knn, labels=['Positive'], average=None)
    # false positive
    false_positive_rate = 1 - recall_score(y_test, predicted_sentiments_knn, labels=['Negative'], average=None)
    # confusion
    confusion = confusion_matrix(y_test, predicted_sentiments_knn)
    # classification report
    report = classification_report(y_test, predicted_sentiments_knn, target_names=['Negative', 'Neutral', 'Positive'])
    # sensitivity analysis
    sensitivity_covariate_key = "Change_in_Covariate_KNN"  # Unique key for this widget
    sensitivity_covariate = st.sidebar.number_input("Change in Covariate-KNN(e.g., 0.1 for 10% change):", 0.0, 1.0, 0.1,key=sensitivity_covariate_key)
    
    sensitivity_covariate = str(sensitivity_covariate)
    sensitivity_prediction = knn_model.predict(tfidf_vectorizer.transform([sensitivity_covariate]))

    return comments_df, accuracy_knn, precision_knn, recall_knn, f1_knn, jaccard_knn, gini_positive_knn, gini_negative_knn, true_positive_rate, false_positive_rate, confusion, report, sensitivity_covariate, sensitivity_prediction

st.title("YouTube Movie Trailer Comment Sentiment Analysis")
st.write("In today's time, movie reviews are often biased or incorrect. Movies that the crowd adores may be "
         "considered a flop by critics. David Fincher's Fight Club was declared a poor watch with too much style"
         "over substance. Pixar and Disney's newest movie Elemental touched the heart of adults and children alike but "
         "critics call it a massive failure.")
st.write("YouTube's comment section is widely appreciated for how it enables a viewer to express their feelings about "
         "a certain movie trailer or video. By performing sentiment analysis targeted towards movie trailers, "
         "other users can make informed decisions about their box office choices. Another use case includes "
         "production houses who can understand whether their movie will be a hit or a flop based on the opinion and "
         "expressions of their viewers before the release date.")
youtube_link = st.text_input("Enter a YouTube video link:")

if youtube_link:
    try:
        video_title = get_video_title(youtube_link)
        if video_title:
            st.write(f"Video Title: {video_title}")

        comments_df_LR, accuracy_LR, precision_LR, recall_LR, f1_LR, jaccard_LR, gini_positive_lr, gini_negative_lr, true_positive_rate_lr, false_positive_rate_lr, confusion_lr, report_lr, sensitivity_covariate_lr, sensitivity_prediction_lr = LR(
            youtube_link, num_comments=1000)
        comments_df_KNN, accuracy_KNN, precision_KNN, recall_KNN, f1_KNN, jaccard_KNN, gini_positive_knn, gini_negative_knn, true_positive_rate_knn, false_positive_rate_knn, confusion_knn, report_knn, sensitivity_covariate_knn, sensitivity_prediction_knn = KNN(
            youtube_link, num_comments=1000)

        # Display Word Cloud for LR Comments
        comments_lr = comments_df_LR['Comment']
        st.subheader("Word Cloud for Comments")
        create_wordcloud(comments_lr)

        # Calculate the percentages of sentiments
        total_comments_lr = len(comments_df_LR)
        positive_count_lr = (comments_df_LR['Sentiment'] == 'Positive').sum()
        negative_count_lr = (comments_df_LR['Sentiment'] == 'Negative').sum()
        neutral_count_lr = (comments_df_LR['Sentiment'] == 'Neutral').sum()

        positive_percentage_lr = (positive_count_lr / total_comments_lr) * 100
        negative_percentage_lr = (negative_count_lr / total_comments_lr) * 100
        neutral_percentage_lr = (neutral_count_lr / total_comments_lr) * 100

        st.write("Using LR Model:")
        st.write(f"Percentage of Positive Comments: {positive_percentage_lr:.2f}%")
        st.write(f"Percentage of Negative Comments: {negative_percentage_lr:.2f}%")
        st.write(f"Percentage of Neutral Comments: {neutral_percentage_lr:.2f}%")

        with st.sidebar:
            st.write("Logistic Regression is a popular algorithm for text classification due to its simplicity, "
                     "interpretability, and versatility for binary and multiclass classification tasks.")
            st.subheader("METRICS FOR Logistic Regression:")
            st.write(f"Accuracy: {accuracy_LR:.2f}")
            st.write(f"Precision: {precision_LR:.2f}")
            st.write(f"Recall: {recall_LR:.2f}")
            st.write(f"F1-Score: {f1_LR:.2f}")
            st.write(f"Jaccard's Index: {jaccard_LR:.2f}")
            st.write(f"Gini's positive Index: {gini_positive_lr:.2f}")
            st.write(f"Gini's negative index: {gini_negative_lr:.2f}")
            st.write(f"True positive rate: {true_positive_rate_lr}")
            st.write(f"False positive rate: {false_positive_rate_lr}")
            st.write(f"Predicted Sentiment with Covariate Change: {sensitivity_prediction_lr[0]}")
            st.write(f"Confusion matrix: {confusion_lr}")
            st.write(f"Classification report:\n{report_lr}")

            # Create and display the sentiment pie chart for LR
            fig_lr = create_pie_chart(positive_percentage_lr, negative_percentage_lr, neutral_percentage_lr)
            st.pyplot(fig_lr)

            st.markdown("""-----------------""")

            st.write("K-Nearest Neighbors for text analytics is a popular choice, but it may return lower accuracy "
                     "for sparse text data.")
            st.subheader("METRICS FOR K-Nearest Neighbors:")
            st.write(f"Accuracy: {accuracy_KNN:.2f}")
            st.write(f"Precision: {precision_KNN:.2f}")
            st.write(f"Recall: {recall_KNN:.2f}")
            st.write(f"F1-Score: {f1_KNN:.2f}")
            st.write(f"Jaccard's Index: {jaccard_KNN:.2f}")
            st.write(f"Gini's positive Index: {gini_positive_knn:.2f}")
            st.write(f"Gini's negative index: {gini_negative_knn:.2f}")
            st.write(f"True positive rate: {true_positive_rate_knn}")
            st.write(f"False positive rate: {false_positive_rate_knn}")
            st.write(f"Predicted Sentiment with Covariate Change: {sensitivity_prediction_knn[0]}")
            st.write(f"Confusion matrix: {confusion_knn}")
            st.write(f"Classification report:\n{report_knn}")

            # Create and display the sentiment pie chart for KNN
            fig_knn = create_pie_chart(
                (comments_df_KNN['Sentiment'] == 'Positive').mean() * 100,
                (comments_df_KNN['Sentiment'] == 'Negative').mean() * 100,
                (comments_df_KNN['Sentiment'] == 'Neutral').mean() * 100,
            )
            st.pyplot(fig_knn)

    except Exception as e:
        st.write("An error occurred:", e)
