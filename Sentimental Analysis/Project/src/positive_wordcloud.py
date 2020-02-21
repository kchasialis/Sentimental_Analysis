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

    positive = dictionary.loc[dictionary['sentiment'] >= 1]

    positive_words = []

    for word in positive['word']:
        if word in word_list:
            positive_words.append(word)

    text_pos = ' '.join(word for word in positive_words)

    wordcloud_pos = WordCloud(max_font_size=50, max_words=200, background_color="white").generate(text_pos)

    plt.figure()
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(15,10))
