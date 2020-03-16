dist_matrix = [[0] * n_samples for _ in range(n_samples)]
for i in range(n_samples):
    for j in range(i+1, n_samples):
        curr_dist = Jaccard(df.iloc[i], df.iloc[j])
        dist_matrix[i][j] = curr_dist
        dist_matrix[j][i] = curr_dist
        # Produces the distance matrix
    if i % 50 == 0:
        print(i, dist_matrix[i])
