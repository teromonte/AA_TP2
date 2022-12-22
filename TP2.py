#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for assignment 2
"""
import numpy as np
import matplotlib.pyplot as plt
import tp2_aux as aux
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize

CLUSTERS_MIN = 2
CLUSTERS_MAX = 10

X = aux.images_as_matrix()
labels = np.loadtxt("labels.txt", delimiter=",")
labelled_labels = labels[np.nonzero(labels[:,1])]

# https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrixR = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrixR, axis=0)) / np.sum(contingency_matrixR)

pca = PCA(n_components=6)
X_transformed_PCA = pca.fit_transform(X)

transformer = KernelPCA(n_components=6, kernel='rbf')
X_transformed_KPCA = transformer.fit_transform(X)

ezmbedding = Isomap(n_components=6)
X_transformed_Isomap = ezmbedding.fit_transform(X)

res = np.concatenate((X_transformed_PCA, X_transformed_KPCA, X_transformed_Isomap), axis=1)

resSTD = (res - np.mean(res, axis=0)) / np.std(res, axis=0)

report_k = 6      # from comparisons

res_labelled = resSTD[np.nonzero(labels[:,1])]

fig = plt.figure(figsize=(18, 12))
fig.subplots_adjust(hspace=0.22, wspace=0.25)

metrics = {
    "rand index": [],
    "f1 score": [],
    "recall": [],
    "precision": [],
    "purity": [],
    "kmeans loss": []
  }

k_aux = 1
for m in metrics.keys():
    # plotting one graph
    ax = fig.add_subplot(2, 3, k_aux)
    k_aux += 1
    ax.set_title(m)
    ax.set_xlabel('num of clusters')
    ax.set_ylabel(m)
    metrics[m] = [-1] * (CLUSTERS_MAX - CLUSTERS_MIN + 1)
    
# KMeans ##############################################

for k in range(CLUSTERS_MIN, CLUSTERS_MAX+1):
    array_pos = k - CLUSTERS_MIN
    Kmeans_alg = KMeans(n_clusters=k)
    labels_Kmeans = Kmeans_alg.fit_predict(resSTD)
    prelabelled_Kmeans = labels_Kmeans[np.nonzero(labels[:,1])]
    
    score_randIndex = rand_score(labelled_labels[:,1], prelabelled_Kmeans)
    metrics["rand index"][array_pos] = score_randIndex
    
    score_f1 = f1_score(labelled_labels[:,1], prelabelled_Kmeans, average='macro')
    metrics["f1 score"][array_pos] = score_f1
    
    score_recall = recall_score(labelled_labels[:,1], prelabelled_Kmeans, average='macro')
    metrics["recall"][array_pos] = score_recall

    score_precision = precision_score(labelled_labels[:,1], prelabelled_Kmeans, average='macro')
    metrics["precision"][array_pos] = score_precision

    score_purity = purity_score(labelled_labels[:,1], prelabelled_Kmeans)
    metrics["purity"][array_pos] = score_purity
    
    metrics["kmeans loss"][array_pos] = Kmeans_alg.inertia_
    
for ax in fig.get_axes():
    ax.plot( range(CLUSTERS_MIN, CLUSTERS_MAX+1), metrics[ax.get_title()], 
            color='green', label='KMeans') 
    
aux.report_clusters(labels[:,0], KMeans(n_clusters=report_k).fit_predict(resSTD), "report_kmeans.html")

# AgglomerativeClustering #############################

for k in range(CLUSTERS_MIN, CLUSTERS_MAX+1):
    array_pos = k - CLUSTERS_MIN
    labels_Agg = AgglomerativeClustering(n_clusters=k).fit_predict(resSTD)
    prelabelled_Agg = labels_Agg[np.nonzero(labels[:,1])]
    
    score_randIndex = rand_score(labelled_labels[:,1], prelabelled_Agg)
    metrics["rand index"][array_pos] = score_randIndex
    
    score_f1 = f1_score(labelled_labels[:,1], prelabelled_Agg, average='macro')
    metrics["f1 score"][array_pos] = score_f1
    
    score_recall = recall_score(labelled_labels[:,1], prelabelled_Agg, average='macro')
    metrics["recall"][array_pos] = score_recall

    score_precision = precision_score(labelled_labels[:,1], prelabelled_Agg, average='macro')
    metrics["precision"][array_pos] = score_precision

    score_purity = purity_score(labelled_labels[:,1], prelabelled_Agg)
    metrics["purity"][array_pos] = score_purity
    
for ax in fig.get_axes():
    ax.plot( range(CLUSTERS_MIN, CLUSTERS_MAX+1), metrics[ax.get_title()], 
            color='orange', label='Agglomerative') 
    
aux.report_clusters(labels[:,0], AgglomerativeClustering(n_clusters=report_k).fit_predict(resSTD), 
                    "report_agg.html")


# SpectralClustering ##################################

for k in range(CLUSTERS_MIN, CLUSTERS_MAX+1):
    array_pos = k - CLUSTERS_MIN
    labels_Spectral = SpectralClustering(n_clusters=k, assign_labels='cluster_qr'
                                         ).fit_predict(resSTD)
    prelabelled_Spectral = labels_Spectral[np.nonzero(labels[:,1])]
    
    score_randIndex = rand_score(labelled_labels[:,1], prelabelled_Spectral)
    metrics["rand index"][array_pos] = score_randIndex
    
    score_f1 = f1_score(labelled_labels[:,1], prelabelled_Spectral, average='macro')
    metrics["f1 score"][array_pos] = score_f1
    
    score_recall = recall_score(labelled_labels[:,1], prelabelled_Spectral, average='macro')
    metrics["recall"][array_pos] = score_recall

    score_precision = precision_score(labelled_labels[:,1], prelabelled_Spectral, average='macro')
    metrics["precision"][array_pos] = score_precision

    score_purity = purity_score(labelled_labels[:,1], prelabelled_Spectral)
    metrics["purity"][array_pos] = score_purity
    
for ax in fig.get_axes():
    ax.plot( range(CLUSTERS_MIN, CLUSTERS_MAX+1), metrics[ax.get_title()], 
            color='red', label='Spectral') 
    ax.legend(['KMeans', 'Agglomerative', 'Spectral'])
    
aux.report_clusters(labels[:,0], SpectralClustering(n_clusters=report_k, assign_labels='cluster_qr'
                                                    ).fit_predict(resSTD), "report_spec.html")

# kmeans loss plot
ax.plot( range(CLUSTERS_MIN, CLUSTERS_MAX+1), metrics["kmeans loss"], color='green')

#purity better closer to 1
#precision better closer to 1
#recall better closer to 1
#f1 measure better closer to 1
#randInxex better closer to 1
#for kmeansLoss, the lower the better