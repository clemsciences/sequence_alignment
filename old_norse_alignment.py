
from deterministic_alignment import align_needleman_wunsch
import numpy


VOWELS = ["a", "e", "i", "o", "ǫ", "ö", "ø", "u", "y", "á", "æ", "œ", "é", "í", "ó", "ú", "ý"]
CONSONANTS = [
    "b",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "x",
    "z",
    "þ",
    "ð"]
CHARACTERS = VOWELS + CONSONANTS

human_matrix = -1 * numpy.ones((len(CHARACTERS), len(CHARACTERS))) + \
                   2 * numpy.eye(len(CHARACTERS))


def align(ns, gs, np):
    ali1, ali2, ali3 = align_needleman_wunsch(ns, gs, human_matrix, 5, CHARACTERS)
    print("", "".join(ali1), "\n", "".join(ali2), "\n", "".join(ali3))
    print("")

    ali1, ali2, ali3 = align_needleman_wunsch(ns, np, human_matrix, 5, CHARACTERS)
    print("", "".join(ali1), "\n", "".join(ali2), "\n", "".join(ali3))
    print("")

    ali1, ali2, ali3 = align_needleman_wunsch(np, gs, human_matrix, 5, CHARACTERS)
    print("", "".join(ali1), "\n", "".join(ali2), "\n", "".join(ali3))


align("ketill", "ketils", "katlar")
