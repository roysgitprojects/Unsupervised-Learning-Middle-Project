from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA
import numpy as np

import data_set_preparations

import clustering
import fit_to_external_classification
import predict_nuber_of_clusters


def main():
    """
    Main function. Clusters the data and compare between clustering methods.
    Please note that in order to avoid a lot of figures on the screen,
    the next figure won't appear until the current figure is closed.
    :return: None
    """
    # read and prepare the data
    data = data_set_preparations.prepare_data_set(3)
    # scale the data
    data = data_set_preparations.scale_the_data(data)
    # normalize the data
    data = normalize(data)
    # reduce dimension to 2d
    points = perform_pca(data)

    # number of real labels
    print('data 1 real labels', len(np.unique(fit_to_external_classification.get_real_labels(1))))
    print('data 2 real labels', len(np.unique(fit_to_external_classification.get_real_labels(2))))
    print('data 3 real labels', len(np.unique(fit_to_external_classification.get_real_labels(3))))

    predict_nuber_of_clusters.perform_elbow_method(points, 'K means')
    predict_nuber_of_clusters.perform_elbow_method(points, 'Hierarchical')
    predict_nuber_of_clusters.perform_silhouette_method(points, 'GMM')
    predict_nuber_of_clusters.perform_silhouette_method(points, 'Fuzzy C Means')
    predict_nuber_of_clusters.perform_silhouette_method(points, 'Spectral')
    clustering.plot_clustering(points, clustering.cluster(points, 4, 'K means'), 'K means')
    clustering.plot_clustering(points, clustering.cluster(points, 4, 'GMM'), 'GMM')
    clustering.plot_clustering(points, clustering.cluster(points, 4, 'Fuzzy C Means'), 'Fuzzy C Means')
    clustering.plot_clustering(points, clustering.cluster(points, 4, 'Hierarchical'), 'Hierarchical')
    clustering.plot_clustering(points, clustering.cluster(points, 4, 'Spectral'), 'Spectral')

    # statistical tests
    # create a dictionary of method and nmi scores list
    algorithms_and_n_clusters = [['K means', 4], ['GMM', 4], ['Fuzzy C Means', 4], ['Spectral', 4]]
    algorithm_nmi_dictionary = {}
    for algorithm, n_clusters in algorithms_and_n_clusters:
        algorithm_nmi_dictionary[algorithm] = fit_to_external_classification.nmi_score(
            fit_to_external_classification.get_real_labels(3), points,
            n_clusters=n_clusters, method=algorithm)
    linkages = ['ward', 'average', 'complete', 'single']
    for linkage in linkages:
        algorithm_nmi_dictionary['Hierarchical' + linkage] = fit_to_external_classification.nmi_score(
            fit_to_external_classification.get_real_labels(3), points,
            n_clusters=4, method='Hierarchical', linkage=linkage)
    print('u test')
    for key1 in algorithm_nmi_dictionary:
        for key2 in algorithm_nmi_dictionary:
            if key1 != key2:
                print('for', key1, 'and', key2, 'p value is',
                      fit_to_external_classification.u_test(algorithm_nmi_dictionary[key1],
                                                            algorithm_nmi_dictionary[key2]))


def perform_pca(data):
    """
    Performs PCA algorithm to 2 dimensions.
    :param data: data to perform the algorithm on
    :return: points after dimension reduction
    """
    pca = PCA(n_components=2)
    points = pca.fit_transform(data)
    return points


if __name__ == '__main__':
    main()
