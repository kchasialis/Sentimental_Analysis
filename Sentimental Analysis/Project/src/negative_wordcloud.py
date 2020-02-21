import matplotlib.pyplot as plt
import pickle
import pandas as pd
from nltk import word_tokenize
from wordcloud import WordCloud

if __name__ == "__main__":

    data = pickle.load(open("train_df.pkl", "rb"))

    sentences = ' '.join(sent for sent in data['tweet'])

    word_list = word_tokenize(sentences)

    dictionary = pd.read_csv(r'../lexica/affin/affin.txt', sep='\t', names=['word', 'sentiment'])

    dictionary = dictionary.sort_values(by=['sentiment'], ascending=True)

    negative = dictionary.loc[dictionary['sentiment'] <= -1]

    negative_words = []

    for word in negative['word']:
        if word in word_list:
            negative_words.append(word)

    text_neg = ' '.join(word for word in negative_words)

    wordcloud_neg = WordCloud(max_font_size=50, max_words=200).generate(text_neg)

    plt.figure()
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(15,10))

