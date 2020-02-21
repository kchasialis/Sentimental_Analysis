import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from nltk import word_tokenize
from collections import Counter

if __name__ == "__main__":

    train_df = pickle.load(open("train_df.pkl", "rb"))

    love_words = ['adorable', 'affection', 'angel', 'caring', 'chocolate', 'compassion', 'dear', 'desire', 'devotion',
                  'forever', 'fondness', 'family', 'friendship', 'happy', 'happiness', 'husband', 'hug', 'hugs',
                  'husband', 'wife', 'joy', 'kiss', 'love', 'sex', 'relationship', 'passion', 'sweet', 'sweetheart',
                  'trust', 'like']

    hate_words = ['hate', 'dislike', 'disgust', 'pain', 'antipathy', 'hatred', 'antagonism', 'repulsion', 'hostility',
                  'horror', 'war', 'abandon', 'terrible', 'awful', 'offensive', 'hideous', 'creepy', 'chaotic',
                  'horrifying']

    political_words = ['trump', 'obama', 'political', 'party', 'vote', 'voting', 'conservative', 'debate', 'campaign',
                       'contribution', 'election', 'government', 'economy', 'poll', 'unemployment', 'rate']

    rows, columns = train_df.shape

    df = pd.DataFrame(index=np.arange(0, rows), columns=['id', 'subject'])

    for index in np.arange(0, rows):

        tweet = train_df['tweet'][index]

        words = word_tokenize(tweet)
        df['id'][index] = train_df['id'][index]

        found_sub = False
        for word in words:
            if word in love_words:
                df['subject'][index] = 'Love'
                found_sub = True
            if word in hate_words:
                df['subject'][index] = 'Hate'
                found_sub = True
            if word in political_words:
                df['subject'][index] = 'Politics'
                found_sub = True

        if found_sub == False:
            df['subject'][index] = 'N/D'


    df = df.loc[df['subject'] != 'N/D']

    counter = Counter(df['subject'])
    subject = counter.keys()
    subject_counts = counter.values()

    indexes = np.arange(len(subject))
    plt.bar(indexes, subject_counts)
    plt.xticks(indexes, subject)
    plt.show()
