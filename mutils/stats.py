


def adjust_alpha(pval, p_FDR=0.05):
    """
    adjust the significance level to account for multiple testing

    uses the Benjamini Hochberg  method (false discovery rate FDR)

    Parameters
    ----------
    pval : ndarray
        Array with p-values obtained by a statistical test.
    p_FDR : float
        p value controlling the false discovery rate

    """
    import numpy as np

    print("Warning: don't use this code: use statsmodels instead")

    # size of array
    n = float(pval.size)
    # 1:n
    i = np.arange(n) + 1
    # p values sorted in ascending order
    pval_sorted = np.sort(pval, axis=None)
    # sorted values are sig if they are smaller than p_thresh
    p_thresh = p_FDR * i / n
    any_sig = pval_sorted <= p_thresh

    # np.max([]) errors
    # at least one is significant
    if np.any(any_sig):
        # find the largest that is significant
        max_sig = np.max(pval_sorted[pval_sorted <= p_thresh])
        # all that are smaller are also significant
        return (pval <= max_sig)

    # none is significant
    else: 
        return np.zeros_like(pval, dtype=np.bool)


# =======================================================


def return_time(data):
    """
    return time as in Christoph Frei's script

    """
    import scipy as sp

    n = data.size
    return (n + 1) / (n + 1 - sp.stats.rankdata(data))






