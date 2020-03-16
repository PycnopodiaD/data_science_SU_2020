def hierarchical(dist_matrix, k):

    cluster = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
    cluster.fit(dist_matrix)
    labels = cluster.labels_
    return silhouette_score(dist_matrix, labels, metric='precomputed')
