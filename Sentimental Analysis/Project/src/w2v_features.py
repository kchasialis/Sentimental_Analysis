import gensim
import numpy as np
import pickle
from nltk import word_tokenize


def update_affin(train_df, test_df, vectors):

    affin = open(r"../lexica/affin/affin.txt", 'r')

    tmp_lines = affin.readlines()

    lines = []

    min_val = -4
    max_val = 4

    vectors = np.hstack((vectors, np.zeros((vectors.shape[0], 1), dtype=vectors.dtype)))

    for line in tmp_lines:
        tokens = line.split('\t')
        num = 2.0 * ((float(tokens[1]) - min_val) / (max_val - min_val)) - 1.0
        newline = tokens[0] + '\t' + str(num)
        lines.append(newline)

    current = 0
    for tweet in train_df['tweet']:
        sum = 0.0
        count = 0
        for word in tweet.split():
            for line in lines:
                tokens = line.split('\t')
                if word == tokens[0]:
                    sum += float(tokens[1])
                    count += 1
                    break
        if count != 0:
            vectors[current][vectors.shape[1] - 1] = sum / count
        else:
            vectors[current][vectors.shape[1] - 1] = np.random.random_sample()

        current += 1

    for tweet in test_df['tweet']:
        sum = 0.0
        count = 0
        for word in word_tokenize(tweet):
            for line in lines:
                tokens = line.split('\t')
                if word == tokens[0]:
                    sum += float(tokens[1])
                    count += 1
                    break
        if count != 0:
            vectors[current][vectors.shape[1] - 1] = sum / count
        else:
            vectors[current][vectors.shape[1] - 1] = np.random.random_sample()

        current += 1

    return vectors


def update_valence(train_df, test_df, vectors):

    valence = open(r"../lexica/emotweet/valence_tweet.txt")

    vectors = np.hstack((vectors, np.zeros((vectors.shape[0], 1), dtype=vectors.dtype)))

    lines = valence.readlines()

    current = 0
    for tweet in train_df['tweet']:
        sum = 0.0
        count = 0
        for word in word_tokenize(tweet):
            for line in lines:
                tokens = line.split('\t')
                if word == tokens[0]:
                    sum += float(tokens[1])
                    count += 1
                    break
        if count != 0:
            vectors[current][vectors.shape[1] - 1] = sum / count
        else:
            vectors[current][vectors.shape[1] - 1] = np.random.random_sample()

        current += 1

    for tweet in test_df['tweet']:
        sum = 0.0
        count = 0
        for word in word_tokenize(tweet):
            for line in lines:
                tokens = line.split('\t')
                if word == tokens[0]:
                    sum += float(tokens[1])
                    count += 1
                    break
        if count != 0:
            vectors[current][vectors.shape[1] - 1] = sum / count
        else:
            vectors[current][vectors.shape[1] - 1] = np.random.random_sample()

        current += 1

    return vectors


def update_generic(train_df, test_df, vectors):

    generic = open(r"../lexica/generic/generic.txt", 'r')

    vectors = np.hstack((vectors, np.zeros((vectors.shape[0], 1), dtype=vectors.dtype)))

    lines = generic.readlines()

    current = 0

    for tweet in train_df['tweet']:
        sum = 0.0
        count = 0
        for word in word_tokenize(tweet):
            for line in lines:
                tokens = line.split('\t')
                if word == tokens[0]:
                    sum += float(tokens[1])
                    count += 1
                    break
        if count != 0:
            vectors[current][vectors.shape[1] - 1] = sum / count
        else:
            vectors[current][vectors.shape[1] - 1] = np.random.random_sample()

        current += 1

    for tweet in test_df['tweet']:
        sum = 0.0
        count = 0
        for word in word_tokenize(tweet):
            for line in lines:
                tokens = line.split('\t')
                if word == tokens[0]:
                    sum += float(tokens[1])
                    count += 1
                    break
        if count != 0:
            vectors[current][vectors.shape[1] - 1] = sum / count
        else:
            vectors[current][vectors.shape[1] - 1] = np.random.random_sample()

        current += 1

    return vectors

if __name__ == "__main__":

    train_df = pickle.load(open("train_df.pkl", "rb"))

    test_df = pickle.load(open("test_df.pkl", "rb"))

    all_tweets = []

    for tweet in train_df['tweet']:
        all_tweets.append(tweet)

    for tweet in test_df['tweet']:
        all_tweets.append(tweet)

    tokenized_tweets = []

    for tweet in all_tweets:
        tokenized_tweets.append(word_tokenize(tweet))

    model_size = 300

    model_w2v = gensim.models.Word2Vec(tokenized_tweets, size=model_size, window=5, min_count=2, sg=1, hs=0, negative=10,
                                       workers=4, seed=34)

    model_w2v.train(tokenized_tweets, total_examples=len(tokenized_tweets), epochs=20)

    pickle.dump(model_w2v, open("model_w2v_plot.pkl", "wb"))

    vectors = np.zeros((len(tokenized_tweets), model_size))

    tweet_num = 0
    for tweet in tokenized_tweets[:len(train_df['tweet'])]:
        i = 0
        temp_vec = vectors[tweet_num]
        for word in tweet:
            if word not in model_w2v.wv.vocab:
                rand_vec = 2 * np.random.random_sample(model_size) - 1.0
                temp_vec = np.add(rand_vec, temp_vec)
            else:
                temp_vec = np.add(model_w2v[word], temp_vec)
            i += 1

        if i != 0:
            vectors[tweet_num] = np.divide(temp_vec, i)
        else:
            vectors[tweet_num] = 2 * np.random.random_sample(model_size) - 1.0

        tweet_num += 1

    for tweet in tokenized_tweets[len(train_df['tweet']):]:
        i = 0
        temp_vec = vectors[tweet_num]
        for word in tweet:
            if word not in model_w2v.wv.vocab:
                rand_vec = 2 * np.random.random_sample(model_size) - 1.0
                temp_vec = np.add(rand_vec, temp_vec)
            else:
                temp_vec = np.add(model_w2v[word], temp_vec)
            i += 1

        if i != 0:
            vectors[tweet_num] = np.divide(temp_vec, i)
        else:
            vectors[tweet_num] = 2 * np.random.random_sample(model_size) - 1.0

        tweet_num += 1

    w2v_test_train = vectors

    w2v_test_train = update_affin(train_df, test_df, w2v_test_train)

    w2v_test_train = update_generic(train_df, test_df, w2v_test_train)

    w2v_test_train = update_valence(train_df, test_df, w2v_test_train)

    pickle.dump(w2v_test_train, open("w2v_updated.pkl", "wb"))
