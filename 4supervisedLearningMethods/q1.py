"""In this exercise, you will program a simple, but very performant classifier
called k nearest neighbours classifier.
In k nearest neighors classifier, you have access to a library of reference
cases as data vectors. The class membership for these cases is known.
If you want to classify a data point for which the membership is now known,
you look for the closest data point in the reference set with a chosen distance
 function (such as the Euclidean distance). The results for a one nearest
 neighbour classifier is to classify the data point to that class of the
 closest data point in the reference set.

Write a function one_nearest_neighbor_classifier that takes the
following arguments: data point to be classified, and the reference set of
data points, the class labels of the reference set. Use Euclidean distance
as the distance measure. The arguments are NumPy arrays. Print out the result
of your classification, like "0" or "1"
(as determined by the contents in the labels).
"""
"""
Test 1:
import numpy as np
data = np.array([0.4, 0.76])
reference_vectors = np.array([[0.2, 0.4], [0.3, 0.6], [0.7, 0.3], [0.8, 0.4]])
reference_labels = np.array([0, 0, 1, 1])
one_nearest_neighbor_classifier(data, reference_vectors, reference_labels)
Result 1:
0

distance answer  [0.41182520563948, 0.452990066116245, 0.7118988692223074, 0.8924124606929242]
"""
"""
Test 2:
import numpy as np
data = np.array([0.7, 0.4])
reference_vectors = np.array([[0.2, 0.4], [0.3, 0.6], [0.7, 0.3], [0.8, 0.4]])
reference_labels = np.array([0, 0, 1, 1])
one_nearest_neighbor_classifier(data, reference_vectors, reference_labels)
Result 2:
1

distance answer: [0.4999999999999994, 0.6708203932499368, 0.6782329983125267, 0.6855654600401043]
"""

import numpy as np
def one_nearest_neighbor_classifier(data, reference_vectors, reference_labels):
    dist = np.linalg.norm(data-reference_vectors[0]), np.linalg.norm(data-reference_vectors[1]), np.linalg.norm(data-reference_vectors[2]), np.linalg.norm(data-reference_vectors[3])
    print(dist)
    if dist[0] < dist[1] and dist[0] < dist[2] and dist[0] < dist[3]:
        print(f'{reference_labels[0]}')
    elif dist[1] < dist[0] and dist[1] < dist[2] and dist[1] < dist[3]:
        print(f'{reference_labels[1]}')
    elif dist[2] < dist[0] and dist[2] < dist[1] and dist[2] < dist[3]:
        print(f'{reference_labels[2]}')
    elif dist[3] < dist[0] and dist[3] < dist[2] and dist[3] < dist[1]:
        print(f'{reference_labels[3]}')
