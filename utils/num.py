import numpy as np;

def to_nparrays(*args):
    arrays = [];
    for _list in args:
        arrays.append(np.asarray(_list));
    return zip(arrays);
