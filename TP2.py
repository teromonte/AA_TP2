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

CLUSTERS_MIN = 2
CLUSTERS_MAX = 10

X = aux.images_as_matrix()
labels = np.loadtxt("labels.txt", delimiter=",")

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

fig = plt.figure(figsize=(18, 12))
fig.subplots_adjust(hspace=0.22, wspace=0.25)

metrics = {
    "rankIndex": [],
    "f1 score": [],
    "recall": [],
    "precision": [],
    "purity": []
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
    labels_Kmeans = KMeans(n_clusters=k).fit_predict(resSTD)
    
    score_randIndex = rand_score(labels[:,1], labels_Kmeans)
    metrics["rankIndex"][array_pos] = score_randIndex
    
    score_f1 = f1_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["f1 score"][array_pos] = score_f1
    
    score_recall = recall_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["recall"][array_pos] = score_recall

    score_precision = precision_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["precision"][array_pos] = score_precision

    score_purity = purity_score(labels[:,1], labels_Kmeans)
    metrics["purity"][array_pos] = score_purity
    
for ax in fig.get_axes():
    ax.plot( range(CLUSTERS_MIN, CLUSTERS_MAX+1), metrics[ax.get_title()] ) 
    
best_k = 4 # how to calculate?
aux.report_clusters(labels[:,0], KMeans(n_clusters=best_k).fit_predict(resSTD), "report_kmeans.html")

# AgglomerativeClustering #############################

for k in range(CLUSTERS_MIN, CLUSTERS_MAX+1):
    array_pos = k - CLUSTERS_MIN
    labels_Kmeans = AgglomerativeClustering(n_clusters=k).fit_predict(resSTD)
    
    score_randIndex = rand_score(labels[:,1], labels_Kmeans)
    metrics["rankIndex"][array_pos] = score_randIndex
    
    score_f1 = f1_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["f1 score"][array_pos] = score_f1
    
    score_recall = recall_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["recall"][array_pos] = score_recall

    score_precision = precision_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["precision"][array_pos] = score_precision

    score_purity = purity_score(labels[:,1], labels_Kmeans)
    metrics["purity"][array_pos] = score_purity
    
for ax in fig.get_axes():
    ax.plot( range(CLUSTERS_MIN, CLUSTERS_MAX+1), metrics[ax.get_title()] ) 
    
best_k = 4 # how to calculate?
aux.report_clusters(labels[:,0], AgglomerativeClustering(n_clusters=best_k).fit_predict(resSTD), 
                    "report_kmeans.html")


# SpectralClustering ##################################

for k in range(CLUSTERS_MIN, CLUSTERS_MAX+1):
    array_pos = k - CLUSTERS_MIN
    labels_Kmeans = SpectralClustering(n_clusters=k, assign_labels='cluster_qr').fit_predict(resSTD)
    
    score_randIndex = rand_score(labels[:,1], labels_Kmeans)
    metrics["rankIndex"][array_pos] = score_randIndex
    
    score_f1 = f1_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["f1 score"][array_pos] = score_f1
    
    score_recall = recall_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["recall"][array_pos] = score_recall

    score_precision = precision_score(labels[:,1], labels_Kmeans, average='macro')
    metrics["precision"][array_pos] = score_precision

    score_purity = purity_score(labels[:,1], labels_Kmeans)
    metrics["purity"][array_pos] = score_purity
    
for ax in fig.get_axes():
    ax.plot( range(CLUSTERS_MIN, CLUSTERS_MAX+1), metrics[ax.get_title()] ) 
    
best_k = 4 # how to calculate?
aux.report_clusters(labels[:,0], SpectralClustering(n_cluster=best_k, assign_labels='clusters_qr').fit_predict(resSTD), 
                    "report_kmeans.html")
# examine the performances varying the main parameters of each clustering alg


#internalIndex
#kmeansLoss
#externalIndexes

#purity better closer to 1
#precision better closer to 1
#recall better closer to 1
#f1 measure better closer to 1
#randInxex better closer to 1