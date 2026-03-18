"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import scipy.linalg

N_DIMENSIONS = 10

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Best value of k to use, from experience
    k=6

    # Use all features
    features = np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]

    # Super compact implementation of Euclidean distance
    # Source used(Method 3): https://towardsdatascience.com/optimising-pairwise-euclidean-distance-calculations-using-python-fc020112c984
    x = np.sum(test**2, axis=1)[:, np.newaxis]
    y = np.sum(train**2, axis=1)
    xy = np.dot(test, train.T)
    dist = np.sqrt(x + y - 2*xy)

    # Find the index of the k colsest values
    nearest = np.argsort(dist, axis=1)[:, :k]

    # Use to determine the label for each test sample
    # Source used: https://www.w3resource.com/numpy/manipulation/unique.php
    labels = []
    for i in nearest:
        lables = train_labels[i]
        ideal_labels, count = np.unique(lables, return_counts=True)
        # Find the most likely lable using count
        most_likely_label = ideal_labels[np.argmax(count)]
        labels.append(most_likely_label)

    return labels

def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    # Use the values with the lowest eigenvector to select the traning data
    eigenvectors = np.array(model["eigenvectors"])
    pcatrain_data = np.dot((data - np.mean(data)), eigenvectors)

    return pcatrain_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    # Get the dimentions of the coveriance matrix
    covx = np.cov(fvectors_train, rowvar=0)
    N = covx.shape[0]
    # Return the eigenvectors as column vectors in the matrix
    w, eigenvectors = scipy.linalg.eigh(covx, eigvals=(N - N_DIMENSIONS, N - 1))
    eigenvectors = np.fliplr(eigenvectors)

    # Add the necessary inforamtion to model
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["eigenvectors"] = eigenvectors.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """

    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get data out of the model
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # For now, sellect th elables using the square implementation
    list_of_lables = classify_squares(fvectors_test, model)

    # Implementation I would have used if I had more time to work on this to add additional functionality and accuracy.
    # board_list = []
    # board_size = 64

    # # Add the boards to the board list
    # for i in range(0, len(list_of_lables), board_size):
    #     board = list_of_lables[i:i+board_size] 
    #     board_list.append(board)

    # # Look through the values and classify the king again till the correct king is selecte 
    # # and the other is changed to the correct value
    # for i in range(len(board_list)):
    #     king_count = 0
    #     king_values = []
    #     for j in range(board_size):
    #         if (list_of_lables[board_size * i + j] == 'k'): # and capital K
    #             king_values.append([i,j])
    #             king_count += 1
    #     if (king_count > 1):
    #         model["wrong king"] = True
    #         model["king info"] = board_list[i][j]
    #         classify_squares(fvectors_test, model)

    return list_of_lables