import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


if __name__ == "__main__":

    train_df = pickle.load(open("train_df.pkl", "rb"))

    test_df = pickle.load(open("test_df.pkl", "rb"))

    w2v_train_test = pickle.load(open("w2v_updated.pkl", "rb"))

    w2v_xtrain = w2v_train_test[:len(train_df['tweet'])]

    w2v_xtest = w2v_train_test[len(train_df['tweet']):]

    pos_neg_ytrain = []
    pos_neg_xtrain = []
    pos_neg_xtest = []

    count = 0

    for sentiment in train_df['sentiment']:
        if sentiment == 'positive':
            pos_neg_xtrain.append(w2v_xtrain[count])
            pos_neg_ytrain.append(0)
        elif sentiment == 'negative':
            pos_neg_xtrain.append(w2v_xtrain[count])
            pos_neg_ytrain.append(2)

        count += 1

    pos_neu_ytrain = []
    pos_neu_xtrain = []

    count = 0

    for sentiment in train_df['sentiment']:
        if sentiment == 'positive':
            pos_neu_xtrain.append(w2v_xtrain[count])
            pos_neu_ytrain.append(0)
        elif sentiment == 'neutral':
            pos_neu_xtrain.append(w2v_xtrain[count])
            pos_neu_ytrain.append(1)

        count += 1

    neg_neu_ytrain = []
    neg_neu_xtrain = []

    count = 0

    for sentiment in train_df['sentiment']:
        if sentiment == 'neutral':
            neg_neu_xtrain.append(w2v_xtrain[count])
            neg_neu_ytrain.append(1)
        elif sentiment == 'negative':
            neg_neu_xtrain.append(w2v_xtrain[count])
            neg_neu_ytrain.append(2)

        count += 1

    pos_neg_knn = KNeighborsClassifier(n_neighbors=4)

    pos_neg_knn.fit(pos_neg_xtrain, pos_neg_ytrain)

    pos_neg_pred_train = pos_neg_knn.predict_proba(w2v_xtrain)

    pos_neg_pred_test = pos_neg_knn.predict_proba(w2v_xtest)

    pos_neu_knn = KNeighborsClassifier(n_neighbors=4)

    pos_neu_knn.fit(pos_neu_xtrain, pos_neu_ytrain)

    pos_neu_pred_train = pos_neu_knn.predict_proba(w2v_xtrain)

    pos_neu_pred_test = pos_neg_knn.predict_proba(w2v_xtest)

    neg_neu_knn = KNeighborsClassifier(n_neighbors=4)

    neg_neu_knn.fit(neg_neu_xtrain, neg_neu_ytrain)

    neg_neu_pred_train = pos_neu_knn.predict_proba(w2v_xtrain)

    neg_neu_pred_test = pos_neu_knn.predict_proba(w2v_xtest)

    xtrain = []
    ytrain = []
    xtest = []

    count = 0
    for sentiment in train_df['sentiment']:
        if sentiment == 'positive':
            ytrain.append(0)
        elif sentiment == 'neutral':
            ytrain.append(1)
        elif sentiment == 'negtative':
            ytrain.append(2)

        xtrain.append([pos_neg_pred_train[count][0], pos_neg_pred_train[count][1], pos_neu_pred_train[count][0],
                       pos_neu_pred_train[count][1], neg_neu_pred_train[count][0], neg_neu_pred_train[count][1]])
        count += 1

    count = 0

    for tweet in test_df['tweet']:
        xtest.append([pos_neg_pred_test[count][0], pos_neg_pred_test[count][1], pos_neu_pred_test[count][0],
                      pos_neu_pred_test[count][1], neg_neu_pred_test[count][0], neg_neu_pred_test[count][1]])
        count += 1

    knn = KNeighborsClassifier(n_neighbors=4)

    knn.fit(xtrain, ytrain)

    prediction = knn.predict(xtest)

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

    print(f1_score(ytrue, prediction, average='macro'))
