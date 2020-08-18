"""
Part of speech tagging

"""

from collections import defaultdict
import math
import numpy as np
import nltk

try:
    from nltk.corpus import brown
except LookupError:
    nltk.download("brown")

brown_annotated_sentences = brown.tagged_sents()
# brown.words()
print(brown_annotated_sentences[:5])

np.random.seed(94)

training_proportion = 0.8

corpus = list(brown_annotated_sentences)[:5000]
print(corpus[:5])
shuffled_corpus = corpus
np.random.shuffle(shuffled_corpus)
shuffled_corpus = [word_tag for sentence in shuffled_corpus for word_tag in [("--n--", "--s--")]+[word_tag for word_tag in sentence]][:5000]
prep = [word for word, tag in shuffled_corpus]
print(prep[:10])
training_corpus = shuffled_corpus[:int(training_proportion*len(corpus))]
test_corpus = shuffled_corpus[int(training_proportion*len(corpus)):]

vocab = {word: i for i, word in enumerate(sorted(list({word for word, tag in shuffled_corpus})))}

y = []  # tests


alpha = 0.001

# transition counts

# emission counts

# tag counts
# tags = [tag for word, tag in corpus]
# tag_counts = nltk.FreqDist(tags)


def create_dictionaries(training_corpus):
    """
    Input:
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """

    # initialize the dictionaries using defaultdict
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'
    prev_tag = '--s--'

    # use 'i' to track the line number in the corpus
    i = 0

    # Each item in the training corpus contains a word and its POS tag
    # Go through each word and its tag in the training corpus
    for word_tag in training_corpus:

        # Increment the word_tag count
        i += 1

        # Every 50,000 words, print the word count
        if i % 50000 == 0:
            print(f"word count = {i}")

        # get the word and tag using the get_word_tag helper function (imported from utils_pos.py)
        word, tag = word_tag

        # Increment the transition count for the previous word and tag
        transition_counts[(prev_tag, tag)] += 1

        # Increment the emission count for the tag and word
        emission_counts[(tag, word)] += 1

        # Increment the tag count
        tag_counts[tag] += 1

        # Set the previous tag to this tag (for the next iteration of the loop)
        prev_tag = tag

    return emission_counts, transition_counts, tag_counts


emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus)
states = sorted(tag_counts.keys())
print(f"Number of POS tags (number of 'states'): {len(states)}")
print("View these POS tags (states)")
print(states)


def create_transition_matrix(alpha, tag_counts, transition_counts):
    """
    Input:
        alpha: number used for smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        transition_counts: transition count for the previous word and tag
    Output:
        A: matrix of dimension (num_tags,num_tags)
    """
    # Get a sorted list of unique POS tags
    all_tags = sorted(tag_counts.keys())

    # Count the number of unique POS tags
    num_tags = len(all_tags)

    # Initialize the transition matrix 'A'
    A = np.zeros((num_tags, num_tags))

    # Get the unique transition tuples (previous POS, current POS)
    trans_keys = set(transition_counts.keys())

    # Go through each row of the transition matrix A
    for i in range(num_tags):

        # Go through each column of the transition matrix A
        for j in range(num_tags):

            # Initialize the count of the (prev POS, current POS) to zero
            count = 0

            # Define the tuple (prev POS, current POS)
            # Get the tag at position i and tag at position j (from the all_tags list)
            key = (all_tags[i], all_tags[j])

            # Check if the (prev POS, current POS) tuple
            # exists in the transition counts dictionary
            if key in trans_keys:  # complete this line

                # Get count from the transition_counts dictionary
                # for the (prev POS, current POS) tuple
                count = transition_counts[key]

            # Get the count of the previous tag (index position i) from tag_counts
            count_prev_tag = tag_counts[all_tags[i]]

            # Apply smoothing using count of the tuple, alpha,
            # count of previous tag, alpha, and total number of tags
            A[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)

    return A


A = create_transition_matrix(alpha, tag_counts, transition_counts)


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    """
    Input:
        alpha: tuning parameter used in smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        vocab: a dictionary where keys are words in vocabulary and value is an index.
               within the function it'll be treated as a list
    Output:
        B: a matrix of dimension (num_tags, len(vocab))
    """

    # get the number of POS tag
    num_tags = len(tag_counts)

    # Get a list of all POS tags
    all_tags = sorted(tag_counts.keys())

    # Get the total number of unique words in the vocabulary
    num_words = len(vocab)

    # Initialize the emission matrix B with places for
    # tags in the rows and words in the columns
    B = np.zeros((num_tags, num_words))

    # Get a set of all (POS, word) tuples
    # from the keys of the emission_counts dictionary
    emis_keys = set(list(emission_counts.keys()))

    # Go through each row (POS tags)
    for i in range(num_tags):  # complete this line

        # Go through each column (words)
        for j in range(num_words):  # complete this line

            # Initialize the emission count for the (POS tag, word) to zero
            count = 0

            # Define the (POS tag, word) tuple for this row and column
            key = (all_tags[i], vocab[j])

            # check if the (POS tag, word) tuple exists as a key in emission counts
            if key in emis_keys:  # complete this line

                # Get the count of (POS tag, word) from the emission_counts d
                count = emission_counts[key]

            # Get the count of the POS tag
            count_tag = tag_counts[all_tags[i]]

            # Apply smoothing and store the smoothed value
            # into the emission matrix B for this row and column
            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)
    return B


B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))


def initialize(states, tag_counts, A, B, corpus, vocab):
    """
    Input:
        states: a list of all possible parts-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    """
    # Get the total number of unique POS tags
    num_tags = len(tag_counts)

    # Initialize best_probs matrix
    # POS tags in the rows, number of words in the corpus as the columns
    best_probs = np.zeros((num_tags, len(corpus)))

    # Initialize best_paths matrix
    # POS tags in the rows, number of words in the corpus as columns
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)

    # Define the start token
    s_idx = states.index("--s--")

    # Go through each of the POS tags
    for i in range(num_tags):  # complete this line

        # Handle the special case when the transition from start token to POS tag i is zero
        if A[s_idx, i] == 0:  # complete this line

            # Initialize best_probs at POS tag 'i', column 0, to negative infinity
            best_probs[i, 0] = float('-inf')

        # For all other cases when transition from start token to POS tag i is non-zero:
        else:

            # Initialize best_probs at POS tag 'i', column 0
            # Check the formula in the instructions above
            # print(A[s_idx, i])
            # print(corpus[0])
            # print(vocab[corpus[0]])
            # print(B[i, vocab[corpus[0]]])
            best_probs[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])

    return best_probs, best_paths


best_probs, best_paths = initialize(states, tag_counts, A, B, training_corpus, vocab)


def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    """
    Input:
        A, B: The transition and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    """
    # Get the number of unique POS tags (which is the num of rows in best_probs)
    num_tags = best_probs.shape[0]

    # Go through every word in the corpus starting from word 1
    # Recall that word 0 was initialized in `initialize()`
    for i in range(1, len(test_corpus)):

        # Print number of words processed, every 5000 words
        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))

        # For each unique POS tag that the current word can be
        for j in range(num_tags):  # complete this line

            # Initialize best_prob for word i to negative infinity
            best_prob_i = float('-inf')

            # Initialize best_path for current word i to None
            best_path_i = None

            c = math.log(B[j, vocab[test_corpus[i]]])

            # For each POS tag that the previous word can be:
            for k in range(num_tags):  # complete this line

                # Calculate the probability =
                # best probs of POS tag k, previous word i-1 +
                # log(prob of transition from POS k to POS j) +
                # log(prob that emission of POS j is word i)
                prob = best_probs[k, i - 1] + math.log(A[k, j]) + c

                # check if this path's probability is greater than
                # the best probability up to and before this point
                if prob > best_prob_i:  # complete this line
                    # Keep track of the best probability
                    best_prob_i = prob

                    # keep track of the POS tag of the previous word
                    # that is part of the best path.
                    # Save the index (integer) associated with
                    # that previous word's POS tag
                    best_path_i = k

            # Save the best probability for the
            # given current word's POS tag
            # and the position of the current word inside the corpus
            best_probs[j, i] = best_prob_i

            # Save the unique integer ID of the previous POS tag
            # into best_paths matrix, for the POS tag of the current word
            # and the position of the current word inside the corpus.
            best_paths[j, i] = best_path_i

    return best_probs, best_paths


best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)


def viterbi_backward(best_probs, best_paths, corpus, states):
    """
    This function returns the best path.

    """
    # Get the number of words in the corpus
    # which is also the number of columns in best_probs, best_paths
    m = best_paths.shape[1]

    # Initialize array z, same length as the corpus
    z = [None] * m

    # Get the number of unique POS tags
    num_tags = best_probs.shape[0]

    # Initialize the best probability for the last word
    best_prob_for_last_word = float('-inf')

    # Initialize pred array, same length as corpus
    pred = [None] * m

    ## Step 1 ##

    # Go through each POS tag for the last word (last column of best_probs)
    # in order to find the row (POS tag integer ID)
    # with highest probability for the last word
    for k in range(num_tags):  # complete this line

        # If the probability of POS tag at row k
        # is better than the previously best probability for the last word:
        if best_probs[k, m - 1] > best_prob_for_last_word:  # complete this line

            # Store the new best probability for the lsat word
            best_prob_for_last_word = best_probs[k, m - 1]

            # Store the unique integer ID of the POS tag
            # which is also the row number in best_probs
            z[m - 1] = k

    # Convert the last word's predicted POS tag
    # from its unique integer ID into the string representation
    # using the 'states' dictionary
    # store this in the 'pred' array for the last word
    pred[m - 1] = states[z[m - 1]]

    ## Step 2 ##
    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(m - 1, -1, -1):  # complete this line

        # Retrieve the unique integer ID of
        # the POS tag for the word at position 'i' in the corpus
        pos_tag_for_word_i = np.argmax(best_probs[:, i])

        # In best_paths, go to the row representing the POS tag of word i
        # and the column representing the word's position in the corpus
        # to retrieve the predicted POS for the word at position i-1 in the corpus
        z[i - 1] = pos_tag_for_word_i  # best_probs[pos_tag_for_word_i, i]

        # Get the previous word's POS tag in string form
        # Use the 'states' dictionary,
        # where the key is the unique integer ID of the POS tag,
        # and the value is the string representation of that POS tag
        pred[i] = states[z[i - 1]]

    return pred


pred = viterbi_backward(best_probs, best_paths, prep, states)
for i, j, k in zip(shuffled_corpus[:50], prep[:50], pred[:50]):
    print(i, j, k)


def compute_accuracy(pred, y):
    """
    Input:
        pred: a list of the predicted parts-of-speech
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output:

    """
    num_correct = 0
    total = 0
    # Zip together the prediction and the labels
    for prediction, y in zip(pred, y):
        # Split the label into the word and the POS tag
        word_tag_tuple = y.strip().split("\t")
        print(y, word_tag_tuple, prediction)
        # Check that there is actually a word and a tag
        # no more and no less than 2 items
        if len(word_tag_tuple) != 2:  # complete this line
            continue

            # store the word and tag separately
        word, tag = word_tag_tuple

        # Check if the POS tag label matches the prediction
        if prediction == tag:  # complete this line

            # count the number of times that the prediction
            # and label match
            num_correct += 1

        # keep track of the total number of examples (that have valid labels)
        total += 1
    print(num_correct, total)
    return num_correct / total


# compute_accuracy(pred, y)


def forward():
    pass


def backward():
    pass

def viterbi():
    pass