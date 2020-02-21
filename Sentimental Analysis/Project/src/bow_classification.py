import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":

    train_df = pickle.load(open("train_df.pkl", "rb"))

    test_df = pickle.load(open("test_df.pkl", "rb"))

    bow_vectorizer = CountVectorizer(max_df=1.0, min_df=1, max_features=2000, stop_words='english')

    bow_xtrain = bow_vectorizer.fit_transform(train_df['tweet'])

    bow_xtest = bow_vectorizer.transform(test_df['tweet'])

    ytrain = []

    for sentiment in train_df['sentiment']:
        if sentiment == 'positive':
            ytrain.append(0)
        elif sentiment == 'neutral':
            ytrain.append(1)
        else:
            ytrain.append(2)

    svc = svm.SVC(kernel='linear', C=1, probability=True)

    svc = svc.fit(bow_xtrain, ytrain)

    file_contents = [line.strip('\n') for line in open(r'../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt')]

    ytrue = []

    for line in file_contents:
        sentiment = line.split('\t')
        if sentiment[1] == 'positive':
            ytrue.append(0)
        elif sentiment[1] == 'neutral':
            ytrue.append(1)
        else:
            ytrue.append(2)

    prediction = svc.predict(bow_xtest)

    print("BOW SVC")
    print(f1_score(ytrue, prediction, average='macro'))

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(bow_xtrain, ytrain)

    prediction = svc.predict(bow_xtest)
    print("BOW KNN")
    print(f1_score(ytrue, prediction, average='macro'))