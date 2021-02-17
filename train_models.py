import pandas as pd
import numpy as np
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold

######################simple index#################################

#search space of hyperparameters for Decision tree with simple index encoding
param_space_dt_si = {"max_depth": randint(2, 4),
                "max_features": randint(10, 20),
                "min_samples_leaf": randint(1, 15),
                'min_samples_split': randint(2, 15),
                "criterion": ["gini", "entropy"]}

#search space of hyperparameters for Random Forest with simple index encoding
param_space_rf_si = {'n_estimators': randint(150, 1000),
                'max_features': randint(4, 20),
                'max_depth': randint(1, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 20)
                     }

#hyperparameters chosen for Decision tree with simple index encoding
hyperpar_dt_si = {"max_depth": 3,
                "max_features": 16,
                "min_samples_leaf": 6,
                'min_samples_split': 4,
                "random_state": 0}

#hyperparameters chosen for Random Forests with simple index encoding
hyperpar_rf_si = {'n_estimators': 987,
                'max_features': 12,
                'max_depth': 13,
                'min_samples_split':12,
                'min_samples_leaf':13,
                "random_state": 0 }


def main_si():
    #specify path of training and test dataframes created after running the file timestamp_enc.py
    train_si_path = 'train_si.csv'
    test_si_path = 'test_si.csv'
    df_train_si1 = pd.read_csv(train_si_path)
    df_train_si1 = df_train_si1.iloc[:, 1:]
    df_test_si = pd.read_csv(test_si_path)
    df_test_si = df_test_si.iloc[:, 1:]
    print(df_train_si1.head())

    df_val_si = df_train_si1.tail(int(len(df_train_si1) * 20 / 100))
    print(df_val_si.shape)

    df_train_si = df_train_si1.drop(df_val_si.index)
    print(df_train_si.shape)

    print("\n**************Simple Index encoding***************\n")

    print("\n Decision tree with cross validation: ")
    dt = train(model='decision tree',
               xtrain=df_train_si.drop(['trace_id','label'],axis=1),ytrain=df_train_si['label'],
               hyperpar=hyperpar_dt_si)
    print("\nEvaluation on validation set: \n")
    predictions = predict(classifier=dt, xtest=df_val_si.drop(['trace_id','label'],axis=1))
    print_metrics(predictions, df_val_si['label'])
    print("\nEvaluation on test set: \n")
    predictions = predict(classifier=dt, xtest=df_test_si.drop(['trace_id','label'],axis=1))
    print_metrics(predictions, df_test_si['label'])

    print("\n Decision tree with Random Search CV: ")
    train_randomsearch(model='decision tree',train_data=df_train_si1,param_space=param_space_dt_si)

    df_merge = pd.concat([df_train_si1, df_test_si])
    X, y = df_merge.drop(['trace_id', 'label'], axis=1).values, df_merge['label'].values
    print("\n Decision tree with Stratified K Cross Validation: ")
    stratified_kcv(model='decision tree',X=X,y=y,hyperpar=hyperpar_dt_si)

    print("\n Random Forests with cross validation: ")
    rf = train(model='random forests',
               xtrain=df_train_si.drop(['trace_id','label'],axis=1),ytrain=df_train_si['label'],
               hyperpar=hyperpar_rf_si)

    predictions = predict(classifier=rf,xtest= df_val_si.drop(['trace_id','label'],axis=1))
    print("\nEvaluation on validation set: \n")
    print_metrics(predictions, df_val_si['label'])
    print("\nEvaluation on test set: \n")
    predictions = predict(classifier=rf,xtest= df_test_si.drop(['trace_id','label'],axis=1))
    print_metrics(predictions, df_test_si['label'])

    print("\n Random Forests  with Random Search CV: ")
    train_randomsearch(model='random forests',train_data=df_train_si1,param_space=param_space_rf_si)

    df_merge = pd.concat([df_train_si1, df_test_si])
    X, y = df_merge.drop(['trace_id', 'label'], axis=1).values, df_merge['label'].values
    print("\n Random Forests  with Stratified K Cross Validation: ")
    stratified_kcv(model='random forests',X=X,y=y,hyperpar=hyperpar_rf_si)


###################timestamp encoding##########################

#search space of hyperparameters for Decision tree with timestamp encoding
param_space_dt_ts = {"max_depth": randint(2, 10),
              "max_features": randint(10, 75),
              "min_samples_leaf": randint(1, 15),
              'min_samples_split': randint(5, 15),
              "criterion": ["gini", "entropy"]}

#search space of hyperparameters for Random Forests with timestamp encoding
param_space_rf_ts = {'n_estimators': randint(150, 1000),
                  'max_features': randint(20,110),
                  'max_depth': randint(4,20),
                  'min_samples_split':randint(2,20),
                  'min_samples_leaf':randint(1,20) }

#hyperparameters chosen for Decision Tree with timestamp encoding
hyperpar_dt_ts = {"max_depth": 7,
                "max_features": 62,
                "min_samples_leaf": 4,
                'min_samples_split': 14,
                "random_state": 0}

#hyperparameters chosen for Random Forests with timestamp encoding
hyperpar_rf_ts = {'n_estimators': 541,
                'max_features': 62,
                'max_depth': 7,
                'min_samples_split':19,
                'min_samples_leaf':19,
                "random_state": 0 }


def main_ts():
    train_ts_path = 'train_si_ts.csv'
    test_ts_path = 'test_si_ts.csv'
    df_train1 = pd.read_csv(train_ts_path)
    df_train1 = df_train1.iloc[:, 1:]
    df_test = pd.read_csv(test_ts_path)
    df_test = df_test.iloc[:, 1:]
    print(df_train1.head())
    print(df_train1.loc[:,['event_1','ets_1','ts_1','dy_1','mon_1','h_1','min_1','conc_1']].corr())

    df_train1 = df_train1.drop(['ets_' + str(i + 1) for i in range(20)] +
                               ['min_' + str(i + 1) for i in range(20)] +
                               ['h_' + str(i + 1) for i in range(20)], axis=1)

    df_test = df_test.drop(['ets_' + str(i + 1) for i in range(20)] +
                           ['min_' + str(i + 1) for i in range(20)] +
                           ['h_' + str(i + 1) for i in range(20)], axis=1)

    df_val = df_train1.tail(int(len(df_train1) * 20 / 100))
    print(df_val.shape)
    df_train = df_train1.drop(df_val.index)
    print(df_train.shape)

    print("\n**************Timestamp encoding***************\n")

    print("\n Decision tree with cross validation: ")
    dt = train(model='decision tree',
               xtrain=df_train.drop(['trace_id','label'],axis=1),ytrain=df_train['label'],
               hyperpar=hyperpar_dt_ts)

    print("\nEvaluation on validation set: \n")
    predictions = predict(classifier=dt, xtest=df_val.drop(['trace_id','label'],axis=1))
    print_metrics(predictions, df_val['label'])

    print("\nEvaluation on test set: \n")
    predictions = predict(classifier=dt, xtest=df_test.drop(['trace_id','label'],axis=1))
    print_metrics(predictions, df_test['label'])

    print("\n Decision tree with Random Search CV: ")
    train_randomsearch(model='decision tree',train_data=df_train1,param_space=param_space_dt_ts)

    df_merge = pd.concat([df_train1, df_test])
    X, y = df_merge.drop(['trace_id', 'label'], axis=1).values, df_merge['label'].values
    print("\n Decision tree with Stratified K Cross Validation: ")
    stratified_kcv(model='decision tree',X=X,y=y,hyperpar=hyperpar_dt_ts)

    print("\n Random Forests with cross validation: ")
    rf = train(model='random forests',
               xtrain=df_train1.drop(['trace_id','label'],axis=1),ytrain=df_train1['label'],
               hyperpar=hyperpar_rf_ts)
    predictions = predict(classifier=rf,xtest= df_test.drop(['trace_id','label'],axis=1))
    print_metrics(predictions, df_test['label'])

    print("\n Random Forests  with Random Search CV: ")
    train_randomsearch(model='random forests',train_data=df_train1,param_space=param_space_rf_ts)

    df_merge = pd.concat([df_train1, df_test])
    X, y = df_merge.drop(['trace_id', 'label'], axis=1).values, df_merge['label'].values
    print("\n Random Forests  with Stratified K Cross Validation: ")
    stratified_kcv(model='random forests',X=X,y=y,hyperpar=hyperpar_rf_ts)

#function to train model
def train(model,xtrain,ytrain,hyperpar):
    if model == 'decision tree':
      dt = DecisionTreeClassifier(**hyperpar)
      dt.fit(xtrain,ytrain)
      return dt
    elif model == 'random forests':
      clf = RandomForestClassifier(**hyperpar)
      clf.fit(xtrain,ytrain)
      return clf

#function to optimize parameters using randomsearchcv
def train_randomsearch(model,train_data,param_space):
    if model == 'decision tree':
        param_space = param_space
        classifier = DecisionTreeClassifier()

    elif model == 'random forests':
        param_space = param_space
        classifier = RandomForestClassifier()

    scv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    classifier_cv = RandomizedSearchCV(classifier, param_space, cv=scv)
    # Fit it to the data
    classifier_cv.fit(train_data.drop(['trace_id', 'label'], axis=1), train_data['label'])
    # Print the tuned parameters and score
    print("Tuned {} Parameters: {}".format(model,classifier_cv.best_params_))
    print("Best score is {}".format(classifier_cv.best_score_))

#function to predict on x features
def predict(classifier,xtest):
    predictions = classifier.predict(xtest)
    return predictions

#function that prints the evaluation measures
def print_metrics(predictions, ytest):
    print('accuracy=%f'%(accuracy_score(ytest,predictions)))
    print('precision=%f'%(precision_score(predictions,ytest)))
    print('recall=%f'%(recall_score(predictions,ytest)))
    print('f-measure=%f' % (f1_score(predictions, ytest)))

#function to train and evaluate model with stratified k fold cross validation
def stratified_kcv(model, X,y,hyperpar):
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1_measure': []}
    cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    if model == 'decision tree':
       classifier = DecisionTreeClassifier(**hyperpar)
    elif model == 'random forests':
       classifier = RandomForestClassifier(**hyperpar)

    for train_index, test_index in cv.split(X, y):
        xtrain, xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        classifier.fit(xtrain, ytrain)
        pred = classifier.predict(xtest)
        scores['accuracy'].append(accuracy_score(ytest, pred))
        scores['precision'].append(precision_score(ytest, pred))
        scores['recall'].append(recall_score(ytest, pred))
        scores['f1_measure'].append(f1_score(ytest, pred))

    print('average accuracy=%f' % (np.mean(scores['accuracy'])))
    print('average precision=%f' % (np.mean(scores['precision'])))
    print('average recall=%f' % (np.mean(scores['recall'])))
    print('average f-measure=%f' % (np.mean(scores['f1_measure'])))


if __name__== "__main__":
  main_si()

if __name__ == "__main__":
    main_ts()