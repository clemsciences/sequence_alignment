"""

"""

import enum
import itertools
import numpy as np


__author__ = "Clément Besnier <clem@clementbesnier.fr>"

roman_alphabet = list('abcdefghijklmnopqrstuvwxyz')


states = ["M", "I", "S"]


def delta(predicate):
    if predicate:
        return 1
    else:
        return 0


class States(enum.IntEnum):
    MATCHING = 0
    INSERTION = 1
    DELETION = 2


def list_to_pairs(l):
    """
    From list(list(str)) ) to list([str, str])

    >>> l = [[1, 2, 3], [45, 46], [98, 99, 100, 101]]
    >>> list_to_pairs(l)
    [[1, 2], [1, 3], [2, 3], [45, 46], [98, 99], [98, 100], [98, 101], [99, 100], [99, 101], [100, 101]]

    :param l:
    :return:
    """
    res = []
    for i in l:
        length_i = len(i)
        for j in range(length_i-1):
            for k in range(j+1, length_i):
                res.append([i[j], i[k]])
    return res
