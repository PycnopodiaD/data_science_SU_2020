def loose_k_means(df, k):

    centers = [list(df.iloc[index]) for index in random_choice(0, len(df)-1, k)]  # initialize k centers
    n_samples, n_features = df.shape[0], df.shape[1]
    labels = [-1] * n_samples
    sum_error_distance = float('inf')
    while True:
        sample_sum = [[0] * n_features for _ in range(k)]  # each center's ALL sample features SUM
        sample_cnt = [0] * k  # each center's sample COUNT
        curr_sum_error_distance = 0
        for sample_index in range(n_samples):
            center_index, curr_error_distance = center_sample_belong(df.iloc[sample_index], centers)
            labels[sample_index] = center_index
            curr_sum_error_distance += curr_error_distance
            sample_sum[center_index] = [a+b for a, b in zip(sample_sum[center_index], df.iloc[sample_index])]
            sample_cnt[center_index] += 1

        for center_index in range(k):
            new_center = [feature_sum / sample_cnt[center_index] for feature_sum in sample_sum[center_index]]
            centers[center_index] = new_center

        f = collections.Counter(labels)
        print([(center_index, f[center_index]) for center_index in range(k)])
        if abs(sum_error_distance - curr_sum_error_distance) < 5:
            print(round(sum_error_distance, 2),
                  round(curr_sum_error_distance, 2),
                  round(abs(sum_error_distance - curr_sum_error_distance), 2))
            break
        sum_error_distance = curr_sum_error_distance

    score = silhouette_score(df, labels, metric='euclidean')
    return score, centers, labels
