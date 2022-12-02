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

X = aux.images_as_matrix()
labels = np.loadtxt("labels.txt", delimiter=",")


pca = PCA(n_components=6)
X_transformed_PCA = pca.fit_transform(X)

transformer = KernelPCA(n_components=6, kernel='rbf')
X_transformed_KPCA = transformer.fit_transform(X)

embedding = Isomap(n_components=6)
X_transformed_Isomap = embedding.fit_transform(X)

res = np.concatenate((X_transformed_PCA, X_transformed_KPCA, X_transformed_Isomap), axis=1)

resSTD = (res - np.mean(res, axis=0)) / np.std(res, axis=0)

# KMeans ##############################################

labels_Kmeans = KMeans(n_clusters=4).fit_predict(resSTD)
aux.report_clusters(labels[:,0], labels_Kmeans, "report_kmeans.html")

# AgglomerativeClustering #############################

labels_Agg = AgglomerativeClustering(n_clusters = 4).fit_predict(resSTD)
aux.report_clusters(labels[:,0], labels_Agg, "report_agg.html")

# SpectralClustering ##################################

labels_Spec = SpectralClustering(n_clusters=4, assign_labels='cluster_qr').fit_predict(resSTD)
aux.report_clusters(labels[:,0], labels_Spec, "report_spec.html")
