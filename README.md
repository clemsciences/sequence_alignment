
# Sequence alignment

[deterministic_alignment.py](https://github.com/clemsciences/sequence_alignment/blob/master/deterministic_alignment.py)
- Needleman-Wunsch algorithm: global alignment
- Smith-Waterman algorithm: local alignment
- Levenshtein distance: minimum number of operations (insertion, deletion, modification) to transform one sequence to an other sequence

[phmm.py](https://github.com/clemsciences/sequence_alignment/blob/master/phmm.py) : Pair Hidden Markov Model implementation


[blast.py](https://github.com/clemsciences/sequence_alignment/blob/master/blast.py) : heuristics to find local alignments in very long chains


[main.py](https://github.com/clemsciences/sequence_alignment/blob/master/main.py) : examples of presented algorithms


[data_retrieval.py](https://github.com/clemsciences/sequence_alignment/blob/master/data_retrieval.py) : word retrieval thanks to nltk

1. Install nltk with # apt-get install nltk
2. Download Swadesh corpus with
```python
>>> import nltk
>>> nltk.download()
```


[utils.py](https://github.com/clemsciences/sequence_alignment/blob/master/utils.py) : some useful functions which are unclassified


TODO: learning to align pairs of sequences with a Pair Hidden Markov Model.
