def check_consistent_length(*arr):
    lengths = [X.shape[0] for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples:"\
           f"{[int(l) for l in lengths]}"
        )
