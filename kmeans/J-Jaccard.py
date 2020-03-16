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
