import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import mannwhitneyu

from clustering import cluster


def get_real_labels(data_set_number):
    """
    Returns the data's real labels
    :param data_set_number: the number of the data set
    :return: list of the real label's
    """
    if data_set_number == 1:
        data = pd.read_csv("dataset/online_shoppers_intention.csv")
        data['Revenue'] = data['Revenue'].astype('category')
        cat_columns = data.select_dtypes(['category']).columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
        labels = np.array(data['Revenue'])
        return labels
    elif data_set_number == 2:
        data = pd.read_csv("dataset/diabetic_data.csv", skiprows=lambda x: x % 4 != 0)
        # replace ? with None values
        data = data.replace({'?': None})
        # strings to ints
        for column in data.columns:
            if data.dtypes[column] == 'object':
                data[column] = data[column].astype('category')
        cat_columns = data.select_dtypes(['category']).columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
        # impute None with median
        for column in data.columns:
            data.loc[data[column].isnull(), column] = data[column].median()
        # race or gender is the class
        labels = np.array(data['race'])
        return labels
    elif data_set_number == 3:
        data = pd.read_csv("dataset/e-shop clothing 2008.csv", sep=';', skiprows=lambda x: x % 10 != 0)
        data['country'] = data['country'].astype('category')
        cat_columns = data.select_dtypes(['category']).columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
        labels = np.array(data['country'])
        return labels


def nmi_score(labels_true, points, n_clusters, method, linkage='ward'):
    """
    Returns a list with 20 nmi scores.
    :param labels_true: the real labels
    :param points: the points to cluster
    :param n_clusters: the number of clusters
    :param method: clustering method
    :param linkage: if the method is Hierarchical than linkage represents the sub method
    :returns: a list with 20 nmi scores
    """
    score = []
    for i in range(0, 20):
        labels_pred = cluster(points, n_clusters, method, linkage)
        score.append(normalized_mutual_info_score(labels_true, labels_pred))
    return score


def u_test(scores_method_1, scores_method2):
    """
    Returns P value. if p<<0.05 the first scores better than the second
    :param scores_method_1: first method's scores
    :param scores_method2: second method's scores
    :returns: p value
    """
    mann_whitneyu = mannwhitneyu(scores_method_1, scores_method2, alternative='greater')
    # if p value<0.05 than we can say nmi1>nmi2. Therefore, clustering method 1 is better than 2.
    return mann_whitneyu.pvalue


if __name__ == '__main__':
    print(u_test([2 + 1 / i for i in range(1, 20)], [1 + 1 / i for i in range(1, 20)]))
