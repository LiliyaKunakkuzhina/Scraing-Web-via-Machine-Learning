"""

    1)  Preprocessing dataset
    2)  Construction and analyse predictif using Random Forest classifier

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
import sklearn.feature_selection as sfs
from scipy.stats import mstats
from sklearn.feature_selection import RFE

np.random.seed(41)


def get_data_undersampling(data):
    """ Apply undersampling method to dataset used to adjust the class distribution of a dataset """

    sample_size = sum(data.y == 'CEML__PRICE')
    # noisy undersampling
    noisy_indices = data[data.y == 'NOISY'].index
    noisy_sample = data.loc[np.random.choice(noisy_indices, sample_size, replace=False)]
    # ceml_description undersampling
    ceml_description_indices = data[data.y == 'CEML__DESCRIPTION'].index
    ceml_description_sample = data.loc[np.random.choice(ceml_description_indices, sample_size, replace=False)]
    # ceml page description list items undersampling
    ceml_page_desc_indices = data[data.y == 'CEML__PAGE__DESCRIPTION__LIST__ITEMS'].index
    ceml_page_desc_sample = data.loc[np.random.choice(ceml_page_desc_indices, sample_size, replace=False)]
    # ceml price undersampling
    ceml_title_indices = data[data.y == 'CEML__TITLE'].index
    ceml_title_sample = data.loc[np.random.choice(ceml_title_indices, sample_size, replace=False)]
    ceml_price_sample = data.loc[data.y == 'CEML__PRICE']
    # concatenate samples of each class
    list_sample = []
    list_sample.append(noisy_sample)
    list_sample.append(ceml_page_desc_sample)
    list_sample.append(ceml_description_sample)
    list_sample.append(ceml_price_sample)
    list_sample.append(ceml_title_sample)
    data = pd.concat(list_sample)
    X = data.values[:, 0:14]
    Y = data.values[:, 14]

    return X, Y


def get_data_oversampling(X_train, y_train):
    """ Apply oversampling method to dataset used to adjust the class distribution of a dataset """

    ros = RandomOverSampler(random_state=0)
    x_oversampled, y_oversampled = ros.fit_sample(X_train, y_train)
    X_train = x_oversampled
    y_train = y_oversampled

    return X_train, y_train







def get_split_data(X, Y):
    """ Spliting dataset into train and test dataset """

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    return X_train, X_test, y_train, y_test

def get_preprocess_dataset(data):
    """ Generalization of the analysis preprocess """

    # Get dataset with only textual tag
    tag_text_list = ["p", "div", "label", "tr", "th", "b", "span", "strong", "title", "td", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "mark", "em"]
    data = data[data.tag_name.apply(lambda x: True if x in tag_text_list else False)]

    data = data.drop(['tag_name'], axis=1)
    # Factor of the dummies features
    features = data.columns.values.tolist()
    features.remove('inner_text_length')
    features.remove('child_text_length')
    # Distribution of the variable Y
    sns.set(style="darkgrid")
    sns.countplot(y="y", data=data)
    plt.title('Distribution of the variable Y')
    plt.tight_layout()
    plt.margins()
    # plt.show()
    # Boxplot of the features 'inner_text_length'
    # data.boxplot(column='inner_text_length', by='y')
    # plt.tight_layout()
    # plt.ylim(0, 400)
    # plt.show()
    # Boxplot of the features 'child_text_length'
    # data.boxplot(column='child_text_length', by='y')
    # plt.tight_layout()
    # plt.ylim(0, 100)
    # plt.show()
    # Distribution of the dummies features
    # categorical_data = data.drop(['inner_text_length', 'child_text_length', 'y'], axis=1)
    # hist = categorical_data.hist()
    # pl.suptitle('Distribution of the dummies variables')
    # plt.show()
    # Elimination of dummies features with missing staff
    data = data.drop(['is_sib_a', 'is_sib_input', 'is_desc_comment', 'is_desc_aside', 'is_desc_menu', 'contains_rights_reserved', 'contains_like', 'contains_share', 'is_link'], axis=1)

    return data


def get_test_kw_inner_text_length_y(data):
    """ Test non parametric Kruskal-Wallis between inner_text_length and Y """

    title = data[data['y'].apply(lambda x: True if x in "CEML__TITLE" else False)].inner_text_length
    price = data[data['y'].apply(lambda x: True if x in "CEML__PRICE" else False)].inner_text_length
    desc = data[data['y'].apply(lambda x: True if x in "CEML__DESCRIPTION" else False)].inner_text_length
    list = data[data['y'].apply(lambda x: True if x in "CEML__DESCRIPTION__LIST__ITEMS" else False)].inner_text_length
    noisy = data[data['y'].apply(lambda x: True if x in "CEML__NOISY" else False)].inner_text_length
    sample_size = round(len(noisy) / 2)
    title = np.random.choice(title, sample_size)
    desc = np.random.choice(desc, sample_size)
    list = np.random.choice(list, sample_size)
    price = np.random.choice(price, sample_size)
    noisy = np.random.choice(noisy, sample_size)
    M = np.transpose(np.array([title, price, desc, list, noisy]))
    M = pd.DataFrame(M, columns=['CEML__TITLE', 'CEML__PRICE', 'CEML__DESCRIPTION', 'CEML__DESCRIPTION__LIST__ITEMS', 'CEML__NOISY'])
    H, pval = mstats.kruskalwallis(M['CEML__TITLE'].tolist(), M['CEML__PRICE'].tolist(), M['CEML__DESCRIPTION'].tolist(), M['CEML__DESCRIPTION__LIST__ITEMS'].tolist(), M['CEML__NOISY'].tolist())
    print("H-statistic:", H)
    print("P-Value:", pval)
    if pval < 0.05:
        print("Reject NULL hypothesis - Significant differences exist between groups.")
    if pval > 0.05:
        print("Accept NULL hypothesis - No significant difference between groups.")

    return data


def get_test_kw_child_text_length_y(data):
    """ Test non parametric Kruskal-Wallis between child_text_length and Y """

    title = data[data['y'].apply(lambda x: True if x in "CEML__TITLE" else False)].child_text_length
    price = data[data['y'].apply(lambda x: True if x in "CEML__PRICE" else False)].child_text_length
    desc = data[data['y'].apply(lambda x: True if x in "CEML__DESCRIPTION" else False)].child_text_length
    list = data[data['y'].apply(lambda x: True if x in "CEML__DESCRIPTION__LIST__ITEMS" else False)].child_text_length
    noisy = data[data['y'].apply(lambda x: True if x in "CEML__NOISY" else False)].child_text_length
    sample_size = len(noisy)
    title = np.random.choice(title, sample_size)
    desc = np.random.choice(desc, sample_size)
    list = np.random.choice(list, sample_size)
    price = np.random.choice(price, sample_size)
    M = np.transpose(np.array([title, price, desc, list, noisy]))
    M = pd.DataFrame(M, columns=['CEML__TITLE', 'CEML__PRICE', 'CEML__DESCRIPTION', 'CEML__DESCRIPTION__LIST__ITEMS', 'CEML__NOISY'])
    H, pval = mstats.kruskalwallis(M['CEML__TITLE'].tolist(), M['CEML__PRICE'].tolist(), M['CEML__DESCRIPTION'].tolist(), M['CEML__DESCRIPTION__LIST__ITEMS'].tolist(), M['CEML__NOISY'].tolist())
    print("H-statistic:", H)
    print("P-Value:", pval)
    if pval < 0.05: print("Reject NULL hypothesis - Significant differences exist between groups.")
    if pval > 0.05: print("Accept NULL hypothesis - No significant difference between groups.")

    return data


def get_analyse_multiple(data):

    # Test non parametric Kruskal-Wallis between inner_text_length and Y
    get_test_kw_inner_text_length_y(data)
    # Test non parametric Kruskal-Wallis between child_text_length and Y
    get_test_kw_child_text_length_y(data)
    X = data.values[:, 0:17]
    Y = data.values[:, 17]
    # Features selection
    # Features selection via test Chi2
    chi2_data = get_feature_selection_test_chi2(X, Y, data)
    # Features selection via RFE
    fi_data = get_feature_selection_fi(X, Y, data)
    # Features selection via RFE
    rfe_data = get_feature_selection_rfe(X, Y, data)
    # Elimination of dummies features with missing staff
    data = data.drop(['is_desc_div', 'is_desc_ol', 'is_desc_ad'], axis=1)

    return data


def get_feature_selection_test_chi2(X, Y, data):

    # Features selection via test Chi2
    test = sfs.SelectKBest(score_func=sfs.chi2
                           , k=5)
    fit = test.fit(X, Y)
    chi2_data = pd.DataFrame(columns=['Features', 'Chi2_Features_Extraction'])
    chi2_data.Features = data.columns.values.tolist()
    chi2_data = chi2_data[chi2_data.Features.apply(lambda x: False if x in ['y'] else True)]
    chi2_data.Chi2_Features_Extraction = fit.pvalues_
    chi2_data = chi2_data.sort_values('Chi2_Features_Extraction', ascending=True)
    print(chi2_data)

    return chi2_data


def get_feature_selection_fi(X, Y, data):

    # Classifier Random Forest
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, Y)
    # Features selection via RFE
    fi_data = pd.DataFrame(columns=['Features', 'Feature_Importances'])
    fi_data.Features = list(data)
    fi_data = fi_data[fi_data.Features.apply(lambda x: False if x in ['y'] else True)]
    fi_data.Feature_Importances = list(clf.feature_importances_)
    fi_data = fi_data.sort_values('Feature_Importances', ascending=False)
    fi_data.Feature_Importances = [round(elem, 2) for elem in fi_data.Feature_Importances]

    return fi_data


def get_feature_selection_rfe(X, Y, data):

    clf = RFE(RandomForestClassifier(random_state=42), 5)
    clf.fit(X, Y)
    # Recursive Feature Elimination
    rfe_data = pd.DataFrame(columns=['Features', 'RFE'])
    rfe_data.Features = list(data)
    rfe_data = rfe_data[rfe_data.Features.apply(lambda x: False if x in ['y'] else True)]
    rfe_data.RFE = list(clf.support_)
    rfe_data = rfe_data.sort_values('RFE', ascending=False)

    return rfe_data


def get_random_forest_classifier(X_train, X_test, y_train, y_test):
    """ Construction and Training the classifier RandomForest """

    # Construction of classifier
    clf = RandomForestClassifier(random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf)

    return y_test, y_pred


def get_analyse_metric(y_test, y_pred):
    """ Get analyse metric for evaluate the classifier """

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    print("Accuracy is ", accuracy)
    print("Precision is ", precision)
    print("Recall is ", recall)
    print("F1 is ", f1)
    metric_data = pd.DataFrame(columns=['Metrics', 'Values'])
    metric_data.Values = [accuracy, precision, recall, f1]
    metric_data.Metrics = ["Accuracy", "Precision", "Recall", "F1"]
    print(metric_data)

    return metric_data


def to_train_random_forest_classifier(X_train, X_test, y_train, y_test):
    """ Construction and Training the classifier RandomForest """

    # RandomForest generation with hyperparameters
    rfc = RandomForestClassifier(random_state=0)
    param_grid = { 'n_estimators': [5, 7], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy'], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False] }
    clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf)

    return y_test, y_pred


def get_and_fit_classifier(data):
    X = data.values[:, 0:14]
    Y = data.values[:, 14]

    # Get balanced dataset(with classification homogeneous)
    X, Y = get_data_undersampling(data)
    # Spliting dataset into train and test dataset
    X_train, X_test, y_train, y_test = get_split_data(X, Y)
    # Get balanced dataset(with classification homogeneous)
    X_train, y_train = get_data_oversampling(X_train, y_train)
    # Build classifier RandomForest
    y_test, y_pred = get_random_forest_classifier(X_train, X_test, y_train, y_test)
    # Analyse and synthese of the classification model
    metric_data = get_analyse_metric(y_test, y_pred)
    # Training classifier RandomForest with hyperparemeters
    y_test, y_pred = to_train_random_forest_classifier(X_train, X_test, y_train, y_test)
    metric_data_tr = get_analyse_metric(y_test, y_pred)

    return metric_data, metric_data_tr

def execute_all_analyse_function(data):
    data_preproc = get_preprocess_dataset(data)
    data_analyse = get_analyse_multiple(data_preproc)
    get_and_fit_classifier(data_analyse)


if __name__ == "__main__":
   with open('/home/arnaud/Desktop/ML_process_file/DATA_all.csv') as f:
        data = pd.read_csv(f)
        execute_all_analyse_function(data)

