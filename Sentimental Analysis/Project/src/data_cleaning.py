import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from string import punctuation

def clean_tweet(tweet):

    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = re.sub(r"\d", "", tweet)
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"_+", "", tweet)
    tweet = re.sub('\$', "", tweet)
    tweet = tweet.lower()

    tokens = tweet.split()
    filtered = [w for w in tokens if not w in stopwords.words('english')]

    tweet = ' '.join(word for word in filtered)

    return tweet.strip()


if __name__ == "__main__":

    train_data = r"../twitter_data/train2017.tsv"
    test_data  = r"../twitter_data/test2017.tsv"

    train_df = pd.read_csv(train_data, sep='\t', names=['unknown', 'id', 'sentiment', 'tweet'])

    test_df = pd.read_csv(test_data, sep='\t', names=['unknown', 'id', 'sentiment', 'tweet'])

    train_df['tweet'] = train_df['tweet'].apply(clean_tweet)

    test_df['tweet'] = test_df['tweet'].apply(clean_tweet)

    pickle.dump(test_df, open("test_df.pkl", "wb"))

    pickle.dump(train_df, open("train_df.pkl", "wb"))