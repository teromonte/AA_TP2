#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for assignment 2
"""
import numpy as np
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

# KMeans ##############################################

labels_Kmeans = KMeans(n_clusters=4).fit_predict(resSTD)
aux.report_clusters(labels[:,0], labels_Kmeans, "report_kmeans.html")

aKmeans_randIndex = rand_score(labels[:,1], labels_Kmeans)
bKmeans_f1 = f1_score(labels[:,1], labels_Kmeans, average='macro')
cKmeans_recall = recall_score(labels[:,1], labels_Kmeans, average='macro')
dKmeans_precision = precision_score(labels[:,1], labels_Kmeans, average='macro')
eKmeans_purity = purity_score(labels[:,1], labels_Kmeans)

# AgglomerativeClustering #############################

labels_Agg = AgglomerativeClustering(n_clusters = 4).fit_predict(resSTD)
aux.report_clusters(labels[:,0], labels_Agg, "report_agg.html")

aAgg_randIndex = rand_score(labels[:,1], labels_Agg)
bAgg_f1 = f1_score(labels[:,1], labels_Agg, average='macro')
cAgg_recall = recall_score(labels[:,1], labels_Agg, average='macro')
dAgg_precision = precision_score(labels[:,1], labels_Agg, average='macro')
eAgg_purity = purity_score(labels[:,1], labels_Agg)

# SpectralClustering ##################################

labels_Spec = SpectralClustering(n_clusters=4, assign_labels='cluster_qr').fit_predict(resSTD)
aux.report_clusters(labels[:,0], labels_Spec, "report_spec.html")

aSpec_randIndex = rand_score(labels[:,1], labels_Spec)
bSpec_f1 = f1_score(labels[:,1], labels_Spec, average='macro')
cSpec_recall = recall_score(labels[:,1], labels_Spec, average='macro')
dSpec_precision = precision_score(labels[:,1], labels_Spec, average='macro')
eSpec_purity = purity_score(labels[:,1], labels_Spec)

# examine the performances varying the main parameters of each clustering algo




#internalIndex
#kmeansLoss
#externalIndexes

#purity better closer to 1
#precision better closer to 1
#recall better closer to 1
#f1 measure better closer to 1
#randInxex better closer to 1