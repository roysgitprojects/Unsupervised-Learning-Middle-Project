import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from yellowbrick.cluster import KElbowVisualizer

import clustering


def perform_elbow_method(points, method):
    """
    Perform and visualize elbow method.
    :param points: the data's points
    :param method: clustering method - K means or Hierarchical
    :return: None
    """
    if method == 'K means':
        model = KMeans()
    elif method == 'Hierarchical':
        model = AgglomerativeClustering()
    else:
        raise Exception('This elbow method designed only for K means and Hierarchical')
    visualizer = KElbowVisualizer(model, k=(1, 12))
    # Fit the data to the visualizer
    visualizer.fit(points)
    visualizer.set_title("The Elbow Method")
    visualizer.show()


def perform_silhouette_method(points, method):
    """
    Calculate and visualize silhouette scores
    :param points: data's points
    :param method: clustering method
    :return: None
    """
    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # Create a figure
        fig = plt.figure()
        fig.set_size_inches(18, 7)
        ax = fig.add_subplot(111)
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(points) + (n_clusters + 1) * 10])

        # find the labels for the clustering method and number of clusters
        cluster_labels = clustering.cluster(points, n_clusters, method)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(points, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(points, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("The Silhouette score with %d clusters is %f" % (n_clusters, silhouette_avg)),
                     fontsize=14, fontweight='bold')

    plt.show()
