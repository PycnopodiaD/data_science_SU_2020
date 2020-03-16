def strict_k_means(df, k):

    centers = [list(df.iloc[index]) for index in random_choice(0, len(df)-1, k)]  # initialize k centers
    n_samples, n_features = df.shape[0], df.shape[1]
    labels = [-1] * n_samples
    max_iteration, cur_iteration = 10, 1
    while True:
        cluster_stay_same = True
        sample_sum = [[0] * n_features for _ in range(k)]  # each center's ALL sample features SUM
        sample_cnt = [0] * k  # each center's sample count
        for sample_index in range(n_samples):
            center_index, _ = center_sample_belong(df.iloc[sample_index], centers)
            if labels[sample_index] != center_index:
                cluster_stay_same = False
                labels[sample_index] = center_index
            sample_sum[center_index] = [a+b for a, b in zip(sample_sum[center_index], df.iloc[sample_index])]
            sample_cnt[center_index] += 1

        for center_index in range(k):
            new_center = [feature_sum / sample_cnt[center_index] for feature_sum in sample_sum[center_index]]
            centers[center_index] = new_center

        f = collections.Counter(labels)
        print([(center_index, f[center_index]) for center_index in range(k)])

        if cluster_stay_same or cur_iteration >= max_iteration:
            break
        cur_iteration += 1

    score = silhouette_score(df, labels, metric='euclidean')
    return score, centers, labels
