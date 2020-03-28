"""TakeHomeExam_EricPederson.py 
The purpose of this script is to take the datfile, check if it is binary or not
and then run the appropriate distance calculation on the datafile. For files
that contain binary data, the runJac function will be implemented resulting in
the calcuation of the Jaccard distance. For files that do not conatin binary
data the runEluc function will be implemented resulting in the calcuation of the
Eculidean distance.

To run the script a datafile (.data) is required as an argument. For example;

$ python TakeHomeExam_EricPederson.py measurements.data

This script requires that 'numpy', 'pandas', 'matplotlib' and 'sklearn' be
installed within the Python environment this script is running on.
The other three modules, os, random and collections, should already be installed

This file contains the following functions for the elucidean distance
calculation:

    * dispatch_func - Checks whether the input datafile is binary or not.
    * convert_eluc_data_to_csv - Returns a .csv file.
    * random_choice - Returns a list of randomly selected sample indexes.
    * Eculidean - Returns the eculidean distances.
    * center_sample_belong - Returns the silhouette score, features of the
centers and the labels with respect to the samples.
    * strict_k_means - Returns the silhouette score, features of the centers
and the labels with respect to the dataframe.
    * runEluc - The main function of the script.

This file contains the following functions for the jaccard distance
calculation:

    * dispatch_func - Checks whether the input datafile is binary or not.
    * convert_jac_data_to_csv - Returns a .csv file.
    * Jaccard - Returns the jaccard distances.
    * hierarchical - Returns the silhouette score.
    * runJac - The main function of the script.
"""

#! /usr/bin/python
import os
import sys
import random
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

def convert_eluc_data_to_csv(data_path, csv_path):
    """
    Function that locates the data and converts the non-binary .data 
    format file, into a .csv format file.

    Parameters
    ----------
    data_path : str
        Location of the .data format file
    csv_path : str
        Location of the .csv format file
    Returns:
        None
    ----------
    """
    df = pd.DataFrame(columns=['col' + str(i + 1) for i in range(12)])
    with open(data_path, 'r') as fr:
        while True:
            line = fr.readline()
            if len(line) == 0:
                break
            row = {'col' + str(i + 1): float(val) for i, val in enumerate(line[:-1].split())}
            df = df.append(row, ignore_index=True)
    df.to_csv(csv_path, index=False)

def random_choice(lower, upper, k):
    """
    Function that randomly selects k samples as the initial clustering centers

    Again to accomplish this make three varaibles; lower, upper and k.
    The lower variable  should equal lambda n

    Parameters
    ----------
    lower : int
        The lower limit of the selection integer pool
    upper : int
        The upper limit of the selection integer pool
    k : int
        count of the clusters
    Returns: list
        Randomly selected sample indexes
    ----------
    """
    chosen = set()
    while len(chosen) < k:
        chosen.add(random.randint(lower, upper))
    return list(chosen)


def Eculidean(center, sample):
    """
    Function to calculate the Eculidean distance

    Two varaibles are required, center and sample, which come from the main
    function.

    Parameters
    ----------
    center : list
        The features of one cluster center
    sample : Series
        The features of one sample
    Returns: float
        Eculidean distance between center and sample
    ----------
    """
    return np.sqrt(sum((a - b)*(a - b) for a, b in zip(center, sample)))


def center_sample_belong(sample, centers):
    """
    Function to calculate which cluster that this sample belongs too.

    Two varaibles are required, center and sample, which are used in the
    previous function, def Eculidean.

    Parameters
    ----------
    sample : Series
        The features of one sample
    centers : list of list
        The features of all centers
    Returns: int
        The closest center's index
    Returns: float
        The distance between the closest center and the sample
    ----------
    """
    distances = [Eculidean(center, sample) for center in centers]
    center_index = np.argmin(distances)
    return center_index, distances[center_index]


def loose_k_means(df, k):
    """
    The function clusters samples into k clusters, loose means truncate
    iteration when diff(neighbor sum-error-dist) < threshold

    The two variables come from the main function, where df is the dataframe and
    k is the number of clusters.

    Parameters
    ----------
    df : DataFrame
        The features of all samples
    k: int
        The numbers of clusters
    Returns: float
        The silhouette score (silhouette_score)
    Returns: list of list
        The features of centers
    Returns: list
        The labels of all samples after clustering
    ----------
    """
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


def strict_k_means(df, k):
    """
    A function for clustering the dataframe (df) samples into k clusters.
    Strict means we truncate iteration when neighboring labels remain same.

    Two varaibles are required, center and sample, which are used in the
    previous function, def loose_k_means.

    Parameters
    ----------
    df: DataFrame
        The features of all samples
    k: int
        The number of clusters
    Returns: float
        The silhouette score (silhouette_score)
    Returns: list of list
        The features of centers
    Returns: list
        Labels of all samples after clustering ends
    ----------
    """
    centers = [list(df.iloc[index]) for index in random_choice(0, len(df) - 1, k)]  # initialize k centers
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

def runEluc(data_path):

    """
    The main function of k-means clustering implementation.

    Again the input varaible is the data_path variable that was used in the
    first function.

    Parameters
    ----------
    data_path : str
        The location of the .data format file
    Returns:
        None
    ----------
    """
    csv_path = data_path.split('.')[0] + '.csv'
    if not os.path.exists(csv_path):
        convert_eluc_data_to_csv(data_path, csv_path)
    df = pd.read_csv(csv_path)
    df = df.apply(lambda x: (x - np.mean(x)) / np.std(x))  # standardization

    scores = []  # scores[i] = score for i+2 clusters
    best_score, best_k, best_centers, best_labels = 0., 0, [], []
    for k in range(2, 21):
        # score, centers, labels = strict_k_means(df, k)
        score, centers, labels = loose_k_means(df, k)
        scores.append(score)
        if best_score < score:
            best_score, best_k, best_centers, best_labels = score, k, centers, labels
        print('number of clusters: %d, score: %.6f' % (k, score))

    print('best number of clusters is: %d' % (np.argmax(scores) + 2))
    plt.scatter(list(range(2, len(scores) + 2)), scores)
    plt.xlabel('number of clusters')
    plt.ylabel('silhouette score')
    plt.xticks(list(range(2, 21, 2)))
    plt.title('Score vs Clusters')
    plt.grid(linestyle='--')
    plt.savefig('k-means.png')
    plt.show()


def convert_jac_data_to_csv(data_path, csv_path):
    """
    Function that locates the data and converts the binary .data 
    format file, into a .csv format file.

    Parameters
    ----------
    data_path : str
        Location of the .data format file
    csv_path : str
        Location of the .csv format file
    Returns:
        None
    ----------
    """
    df = pd.DataFrame(columns=['col'+str(i+1) for i in range(12)])
    with open(data_path, 'r') as fr:
        while True:
            line = fr.readline()
            if len(line) == 0:
                break
            row = {'col'+str(i+1): int(val) for i, val in enumerate(line[:-1].split())}
            df = df.append(row, ignore_index=True)
    df.to_csv(csv_path, index=False)


def Jaccard(sample1, sample2):
    """
    Function that calculates the Jaccard distance from the .csv data file that
    was converted from the data file containing binary data in the def
    convert_data_to_csv function.

    The input is two variables, sample1 & sample2, which comes from the main
    function, def run.

    Parameters
    ----------
    sample1 : Series
        Features of one sample
    sample2 : Series
        Features of another sample
    Returns : float
        Jaccard distance between two samples
    ----------
    """
    inter = sum(a & b for a, b in zip(sample1, sample2))
    union = sum(a | b for a, b in zip(sample1, sample2))
    return 1 - inter / union

def hierarchical(dist_matrix, k):
    """
    Function that implements clustering with k clusters.

    To accomplish this two varaibles, dist_matrix & k are required, which
    are produced from the main function, def run.

    Parameters
    ----------
    dist_matrix : list of list
        The distance matrix for all samples
    k : int
        Number of clusters
    Returns : float
        Silhouette score of current clustering implementation
    ----------
    """
    cluster = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
    cluster.fit(dist_matrix)
    labels = cluster.labels_
    return silhouette_score(dist_matrix, labels, metric='precomputed')

def runJac(data_path):
    """
    The main function of hierarchical clustering implementation that takes the
    data, creates a distance matrix using the Jaccard distance calculation,
    scores the results and then plots them.

    Parameters
    ----------
    data_path : str
        Location of the .data format file
    Returns:
        None
    ----------
    """
    csv_path = data_path.split('.')[0] + '.csv'
    if not os.path.exists(csv_path):
        convert_jac_data_to_csv(data_path, csv_path)
    df = pd.read_csv(csv_path)

    n_samples = df.shape[0]
    dist_matrix = [[0] * n_samples for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            curr_dist = Jaccard(df.iloc[i], df.iloc[j])
            dist_matrix[i][j] = curr_dist
            dist_matrix[j][i] = curr_dist
        # Produces the distance matrix
        if i % 50 == 0:
            print(i, dist_matrix[i])

    # scores[i] = score for i+2 clusters
    scores = [hierarchical(dist_matrix, k) for k in range(2, 21)]
    for k, score in enumerate(scores):
        print('number of clusters: %d, score: %.6f' % (k+2, score))
    print('best number of clusters is: %d' % (np.argmax(scores) + 2))


    # Plot the data using matplotlib
    # make the "silhouette_score vs number of cluster" plot
    plt.scatter(list(range(2, len(scores) + 2)), scores)
    plt.xlabel('number of clusters')
    plt.ylabel('silhouette score')
    plt.xticks(list(range(2, 21, 2)))
    plt.title('Score vs Clusters')
    plt.grid(linestyle='--')
    plt.savefig('hierarchical.png')
    plt.show()

def dispatch_func(data_path):
    """
    Function that checks whether the data in the argument (supplied data file)
    is binary or not.

    Parameters
    ----------
    filename : str
        Location of the .data format file
    Returns:
        None
    ----------
    """
    check = np.loadtxt(data_path)
    if np.array_equal(check, check.astype(bool)) == True:
        runJac(data_path)
    else:
        runEluc(data_path)


if __name__ == '__main__':
    data_path = sys.argv[1]
    dispatch_func(data_path)
