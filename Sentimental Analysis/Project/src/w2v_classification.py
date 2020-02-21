import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import f1_score


if __name__ == "__main__":

    train_df = pickle.load(open("train_df.pkl", "rb"))

    test_df = pickle.load(open("test_df.pkl", "rb"))

    w2v_test_train = pickle.load(open("w2v_updated.pkl", "rb"))

    w2v_xtrain = w2v_test_train[:len(train_df['tweet'])]
    w2v_xtest = w2v_test_train[len(train_df['tweet']):]

    ytrain = []

    for sentiment in train_df['sentiment']:
        if sentiment == 'positive':
            ytrain.append(0)
        elif sentiment == 'neutral':
            ytrain.append(1)
        else:
            ytrain.append(2)

    svc = svm.SVC(kernel='linear', C=1, probability=True)

    svc = svc.fit(w2v_xtrain, ytrain)

    file_contents = [line.strip('\n') for line in open(r'../Project/twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt')]

    ytrue = []

    for line in file_contents:
        sentiment = line.split('\t')
        if sentiment[1] == 'positive':
            ytrue.append(0)
        elif sentiment[1] == 'neutral':
            ytrue.append(1)
        else:
            ytrue.append(2)

    prediction = svc.predict(w2v_xtest)

    print("W2V SVC")
    print(f1_score(ytrue, prediction, average='macro'))

    knn = KNeighborsClassifier(n_neighbors=4)

    knn.fit(w2v_xtrain, ytrain)

    prediction = knn.predict(w2v_xtest)

    print("W2V KNN")
    print(f1_score(ytrue, prediction, average='macro'))