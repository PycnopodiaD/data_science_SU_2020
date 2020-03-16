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
    return np.sqrt(np.sum((a-b)*(a-b) for a, b in zip(center, sample)))
