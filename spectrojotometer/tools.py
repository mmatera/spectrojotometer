from __future__ import print_function
import sys

import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def box_ellipse(A, r):
    """
    Determines the minimum edge half-length of the rectangular box 
    containing the ellipsoid defined by  x^t * A^t * A * x - r**2=0, with faces 
    parallel to the coordinate axes.
    """
    A = np.array(A)
    A = A.transpose().dot(A)
    size = len(A)
    widths = []
    for i in range(size):
        setperp = [k for k in range(i)] + [k for k in range(i + 1, size)]
        v = A[i, setperp]
        A22 = A[setperp][:, setperp]
        try:
            Aperpinv = np.linalg.inv(A22)
            gamma = Aperpinv.dot(v)
            gamma = gamma.dot(v)
            widths.append(r / np.sqrt(A[i, i] - gamma))
        except np.linalg.linalg.LinAlgError:
            widths.append(1.0e300)
    return widths


def normalize_configurations(confs):
    nconfs = {}
    for c in confs:
        if float(sum(c)) / len(c) < 0.5:
            lbl = str(sum([i * 2 ** n for n, i in enumerate(c)]))
            nconfs[lbl] = c
        else:
            nc = [1 - s for s in c]
            lbl = str(sum([i * 2 ** n for n, i in enumerate(nc)]))
            nconfs[lbl] = nc
    return [nconfs[lbl] for lbl in nconfs]
