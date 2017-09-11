from __future__ import print_function
from Preprocessor import Preprocessor

from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

ws = np.arange(50, 500, 3.0)

p = Preprocessor()
pos, X = p.preprocess(pca = False)

n_samples = X.shape[0]
# x1 = np.reshape(np.mean(X[:,:1095], axis = 1), (n_samples,1))
# x2 = np.reshape(np.mean(X[:,1095:2190], axis = 1), (n_samples,1))
# x3 = np.reshape(np.mean(X[:,2190:3285], axis = 1), (n_samples,1))
# x4 = np.reshape(np.mean(X[:,3285:], axis = 1), (n_samples,1))

X = np.concatenate((pos, X), axis = 1)
# X = np.concatenate((X, x2), axis = 1)
# X = np.concatenate((X, x3), axis = 1)
# X = np.concatenate((X, x4), axis = 1)

true_label = np.asarray([0,0,0,0,0,0,0,1,2,2,1,2,1,1,2,2,2,2,2,2,1,1,17,4,2,2,2,1,4,4,2,2,1,9,9,4,4,4,4,4,2,2,6,9,4,4,4,6,6,9,2,2,16,15,15,15,15,15,15,12,15,15,15,12,12,12,12,15,12,12,12,12,11,14,10])
# [8, 30], [14,24], [36], [10], [11], [18], [42],[47], [48], [55,56,52], [71,73]
# could be clustered separately.

for w in ws :
    bandwidths = np.arange(w/12.0,  w/8.0, 2.0)
    for bandwidth in bandwidths:
        # Giving weights
        X[:,0] = X[:,0] * float(w)
        X[:,1] = X[:,1] * float(w)
        
        # training meanshift 
        if bandwidth == 0 :
            clusterer = MeanShift()
            print(clusterer.get_params)
        else :
            clusterer = MeanShift(bandwidth = bandwidth)
        cluster_labels = clusterer.fit_predict(X)
        n_clusters = len(np.unique(cluster_labels))

        # Compare true labels and predicted labels and get the score 
        # Score : [-1, 1]
        score = metrics.adjusted_rand_score(true_label, cluster_labels)
        print("bandwidth {}, W {} - score : {}".format(bandwidth, w, score))
        
        if score >= 0.82 :
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1
            ax1.set_xlim([-1, 1])

            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            try :
                silhouette_avg = silhouette_score(X, cluster_labels)
                print("For bandwidth =", bandwidth,
                  "The average silhouette_score is :", silhouette_avg)
            except :
                continue
            
            # Compute the silhouette scores for each sample
            try :
                sample_silhouette_values = silhouette_samples(X, cluster_labels)
            except :
                continue

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(pos[:, 1], pos[:, 0], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # # Labeling the clusters
            # centers = clusterer.cluster_centers_
            # # Draw white circles at cluster centers
            # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            #             c="white", alpha=1, s=200, edgecolor='k')

            # for i, c in enumerate(centers):
            #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
            #                 s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for Longitude")
            ax2.set_ylabel("Feature space for Latitude")

            plt.suptitle(("Silhouette analysis for MeanShift clustering on sample data "
                          "with bandwidth = %f, w = %f" % (bandwidth,w)),
                         fontsize=14, fontweight='bold')
            plt.show()

        # Taking weights away
        X[:,0] = X[:,0] / float(w)
        X[:,1] = X[:,1] / float(w)