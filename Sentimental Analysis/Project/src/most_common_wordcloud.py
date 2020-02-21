import matplotlib.pyplot as plt
import pickle
import pandas as pd
from nltk import word_tokenize
from collections import Counter
from wordcloud import WordCloud


if __name__ == "__main__":

    data = pickle.load(open("train_df.pkl", "rb"))

    counter = Counter(" ".join(data["tweet"]).split()).most_common(150)

    words = []

    for word, count in counter:
        words.append(word)

    text = ' '.join(word for word in words)

    wordcloud = WordCloud(max_font_size=50, max_words=150).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()