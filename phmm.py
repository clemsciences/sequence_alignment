# -*-coding:utf-8-*-

import numpy as np
import numpy.random as nr
from setuptools.command.test import test

from utils import *

__author__ = "Clément Besnier <clemsciences@aol.com>"

alphabet_ger = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                "u", "v", "w", "x", "y", "z", "ø", "ö"]

EPSILON = 0.00000000001


class CoupleEmissionMatrix:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.n = len(alphabet)
        self.mat = nr.rand(self.n * self.n).reshape((self.n, self.n))
        self.mat /= self.mat.sum(axis=1)
        # self.mat = np.zeros((self.n, self.n))

    def get_emission_matrix(self):
        return self.mat

    def set_emission_matrix(self, mat):
        self.mat = mat

    def __getitem__(self, item):
        return self.mat

    def __len__(self):
        return self.n


class SimpleEmissionMatrix:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.n = len(alphabet)
        self.mat = nr.rand(self.n).reshape((self.n,))
        self.mat /= self.mat.sum()
        # self.mat = np.zeros((self.n,))

    def get_emission_matrix(self):
        return self.mat

    def set_emission_matrix(self, mat):
        self.mat = mat

    def __getitem__(self, item):
        return self.mat

    def __len__(self):
        return self.n


class EmissionMatrix:
    def __init__(self, epsilon, delta_, tau_m, tau_is, lambda_):
        self.epsilon = epsilon
        self.delta_ = delta_
        self.tau_m = tau_m
        self.tau_is = tau_is
        self.lambda_ = lambda_
        self.freq_transitions = np.zeros((3, 3))
        # def add_transition(self, l, m):

    def get_values(self):
        return self.epsilon, self.delta_, self.tau_m, self.tau_is, self.lambda_

    def update_epsilon(self):
        nume = self.freq_transitions[States.INSERTION.value, States.INSERTION.value] + \
               self.freq_transitions[States.DELETION.value, States.DELETION.value]
        denomi = np.sum(self.freq_transitions[States.INSERTION.value, :]) + \
                 np.sum(self.freq_transitions[States.DELETION.value, :]) + EPSILON
        self.epsilon = nume / denomi

    def update_delta(self):
        nume = self.freq_transitions[States.MATCHING.value, States.INSERTION.value] + \
               self.freq_transitions[States.MATCHING.value, States.DELETION.value]
        denomi = np.sum(self.freq_transitions[States.MATCHING.value, :]) + EPSILON
        self.delta_ = nume / denomi

    def update_tau_m(self, n_m_fin):
        nume = n_m_fin
        denomi = n_m_fin + np.sum(self.freq_transitions[States.MATCHING.value, :]) + EPSILON
        self.tau_m = nume / denomi

    def update_tau_is(self, n_is_fin):
        nume = n_is_fin
        denomi = n_is_fin + np.sum(self.freq_transitions[States.INSERTION.value, :]) + \
                 np.sum(self.freq_transitions[States.DELETION.value, :]) + EPSILON
        self.tau_is = nume / denomi

    def update_lambda(self):
        nume = self.freq_transitions[States.INSERTION.value, States.DELETION.value] + \
               self.freq_transitions[States.DELETION.value, States.INSERTION.value]
        denomi = np.sum(self.freq_transitions[States.INSERTION.value, :]) + \
                 np.sum(self.freq_transitions[States.DELETION.value, :]) + EPSILON
        self.lambda_ = nume / denomi

    def get_transition_matrix(self):
        # print("matrice de transition")
        # print(np.array([[1-2*self.delta, self.delta, self.delta],
        #                  [1-self.epsilon-self._lambda, self.epsilon, self._lambda],
        #                  [1-self._lambda-self.epsilon, self._lambda, self.epsilon]]))
        return np.array([[1 - 2 * self.delta_, self.delta_, self.delta_],
                         [1 - self.epsilon - self.lambda_, self.epsilon, self.lambda_],
                         [1 - self.lambda_ - self.epsilon, self.lambda_, self.epsilon]])

    def set_freq_transitions(self, mat):
        self.freq_transitions = mat

    def get_final_transition(self):
        return np.array([self.tau_m, self.tau_is, self.tau_is])

    def update_params(self, n_m_fin, n_is_fin):
        self.update_epsilon()
        self.update_delta()
        self.update_tau_m(n_m_fin)
        self.update_tau_is(n_is_fin)
        self.update_lambda()


class PHMMParameters:
    def __init__(self, epsilon, delta_, tau_m, tau_is, lambda_, pseudo_compte, alphabet):
        self.mat_trans = EmissionMatrix(epsilon, delta_, tau_m, tau_is, lambda_)
        self.mat_emi_m = CoupleEmissionMatrix(alphabet)
        self.mat_emi_is = SimpleEmissionMatrix(alphabet)
        self.pseudo_compte = pseudo_compte
        self.alphabet = alphabet

    # def compute_freq_emissions(self, l_couples_alignes):
    #     for phi, chi in l_couples:

    # def get_emission_matrix(self, i_phi, j_khi):
    #     """
    #     p(\khi_i, \psi_j)
    #     :return:
    #     """
    #     nume = self.freq_emission_couple[i_phi, j_khi] + self.pseudo_compte
    #     denomi = np.sum(np.sum(self.freq_emission)) + self.pseudo_compte
    #     return nume / denomi
    #
    # def get_emission_vector(self, carac):
    #     """
    #     p(\khi_i)
    #     :return:
    #     """
    #     nume = self.freq_emission[carac] + self.pseudo_compte
    #     denomi = np.sum(self.freq_emission) + self.pseudo_compte
    #     return nume / denomi

    def update_params(self, n_m_fin, n_is_fin):
        epsilon, delta_, tau_m, tau_is, lambda_ = self.mat_trans.get_values()
        self.mat_trans.update_params(n_m_fin, n_is_fin)
        diff = abs(epsilon - self.mat_trans.epsilon) + abs(delta_ - self.mat_trans.delta_) + \
               abs(tau_m - self.mat_trans.tau_m) + \
               abs(tau_is - self.mat_trans.tau_is) + abs(lambda_ - self.mat_trans.lambda_)
        return diff


def compute_forward(phi, khi, mat_tra, mat_emi_m, mat_emi_is, trans_fin, alphabet):
    alpha = np.zeros((len(phi), len(khi), mat_tra.shape[0]))
    alpha[0, 0, 0] = 1
    for i in range(len(phi)):
        for j in range(len(khi)):
            if i == 0 and j == 0:
                continue
            if i == 0:
                alpha[i, j, 0] = 0
                alpha[i, j, 1] = 0
                alpha[i, j, 2] = mat_emi_is[alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[2, :] * alpha[i, j - 1, :])
            if j == 0:
                alpha[i, j, 0] = 0
                alpha[i, j, 1] = mat_emi_is[alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[2, :] * alpha[i - 1, j, :])
                alpha[i, j, 2] = 0

            else:
                alpha[i, j, 0] = mat_emi_m[alphabet.index(phi[i]), alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[0, :] * alpha[i - 1, j - 1, :])

                alpha[i, j, 1] = mat_emi_is[alphabet.index(phi[i])] * \
                                 np.sum(mat_tra[1, :] * alpha[i - 1, j, :])
                alpha[i, j, 2] = mat_emi_is[alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[2, :] * alpha[i, j - 1, :])
                # print("\\alpha i j", alpha[i, j, :])
                alpha[i, j, :] /= np.sum(alpha[i, j, :]) + EPSILON
    # print("trans_fin", trans_fin)
    # print("alpha_fin", alpha[len(phi)-1, len(khi)-1, :])
    alpha_fin = np.sum(trans_fin * alpha[len(phi) - 1, len(khi) - 1, :])
    return alpha, alpha_fin


def compute_backward(phi, khi, mat_tra, mat_emi_m, mat_emi_is, trans_fin, alphabet):
    beta = np.zeros((len(phi), len(khi), mat_tra.shape[0]))
    beta[len(phi) - 1, len(khi) - 1, :] = trans_fin
    for i in range(len(phi) - 1, 0, -1):
        for j in range(len(khi) - 1, 0, -1):
            for k in States:
                # print(mat_emi_m.shape)
                # print(mat_emi_is.shape)
                # print(mat_emi_m)
                emi = np.array([mat_emi_m[alphabet.index(phi[i]), alphabet.index(khi[j])],
                                mat_emi_is[alphabet.index(phi[i])], mat_emi_is[alphabet.index(khi[j])]])
                print(emi)
                # print(i+1, j+1, i, j)
                print([beta[i, j, 0], beta[i, j - 1, 1], beta[i - 1, j, 2]])
                beta_transition = np.array([beta[i, j, 0], beta[i, j - 1, 1], beta[i - 1, j, 2]])
                # problem, result is bad
                beta[i - 1, j - 1, k] = np.sum((emi * mat_tra[k, :]) * beta_transition)
            beta[i - 1, j - 1, :] /= np.sum(beta[i - 1, j - 1, :])
    return beta


def viterbi(phi, khi, param):
    n = len(phi)
    m = len(khi)

    matches = np.zeros((n + 1, m + 1))
    matching_path = np.zeros((n + 1, m + 1))
    insertions = np.zeros((n + 1, m + 1))
    insertion_path = np.zeros((n + 1, m + 1))
    deletions = np.zeros((n + 1, m + 1))
    deletion_path = np.zeros((n + 1, m + 1))

    matches[0, 0] = 1
    # insertions[:, 0] = 0
    # insertions[0, :] = 0
    # deletions[:, 0] = 0
    # deletions[0, :] = 0
    optimal_path = []
    alphabet = param.alphabet
    final_state = -1
    for i in range(n):
        for j in range(m):
            matches[i, j] = param.mat_emi_m.mat[alphabet.index(phi[i]), alphabet.index(khi[j])] * \
                            np.max([(1 - 2 * param.mat_trans.delta - param.mat_trans.tau_m) * matches[i - 1, j - 1],
                                    (1 - param.mat_trans.epsilon - param.mat_trans.tau_is - param.mat_trans.lambda_) *
                                    insertions[i - 1, j - 1],
                                    (1 - param.mat_trans.epsilon - param.mat_trans.tau_is - param.mat_trans.lambda_) *
                                    deletions[i - 1, j - 1]])
            matching_path[i, j] = np.argmax(
                [(1 - 2 * param.mat_trans.delta - param.mat_trans.tau_m) * matches[i - 1, j - 1],
                 (1 - param.mat_trans.epsilon - param.mat_trans.tau_is -
                  param.mat_trans.lambda_) * insertions[i - 1, j - 1],
                 (1 - param.mat_trans.epsilon - param.mat_trans.tau_is -
                  param.mat_trans.lambda_) * deletions[i - 1, j - 1]])
            # print(param.mat_emi_is[alphabet.index(khi[j])])
            # print(param.mat_trans.delta)
            # print(matches[i-1, j])
            # print(param.mat_trans.epsilon)
            # print(insertions[i-1, j])
            # TODO correct ValueError: setting an array element with a sequence.
            insertions[i, j] = param.mat_emi_is.mat[alphabet.index(khi[j])] * \
                               np.max([param.mat_trans.delta * matches[i - 1, j],
                                       param.mat_trans.epsilon *
                                       insertions[i - 1, j]])
            insertion_path[i, j] = np.argmax([param.mat_trans.delta * matches[i - 1, j],
                                              param.mat_trans.epsilon * insertions[i - 1, j]])
            deletions[i, j] = param.mat_emi_is.mat[alphabet.index(khi[j])] * \
                              np.max([param.mat_trans.delta * matches[i, j - 1],
                                      param.mat_trans.epsilon * deletions[i, j - 1]])
            deletion_path[i, j] = np.argmax([param.mat_trans.delta * matches[i, j - 1],
                                             param.mat_trans.epsilon * deletions[i, j - 1]])

            if i == n - 1 and j == m - 1:
                # end = np.max([param.mat_trans.tau_m * matches[n, m], param.mat_trans.tau_is * insertions[n, m],
                #               param.mat_trans.tau_is * deletions[n, m]])
                final_state = np.argmax(
                    [param.mat_trans.tau_m * matches[n, m], param.mat_trans.tau_is * insertions[n, m],
                     param.mat_trans.tau_is * deletions[n, m]])

    # print(matching_path)
    print(matches)
    # print(insertion_path)
    print(insertions)
    # print(deletion_path)
    print(deletions)
    i = n
    j = m
    state = states[final_state]  # finale_state[0]
    # optimal_path.append(state)
    first_seq = []
    second_seq = []
    while i > 0 and j > 0:
        # print(state)
        if state == "M" and i > 0 and j > 0:
            i -= 1
            j -= 1
            # print(matching_path[i, j])
            state = states[int(matching_path[i, j])]  # [0]]
            first_seq.append(phi[i])
            second_seq.append(khi[j])
        elif state == "I" and i > 0:
            i -= 1
            state = states[int(insertion_path[i, j])]  # [0]]
            first_seq.append(phi[i])
            second_seq.append("-")
        elif state == "S" and j > 0:
            j -= 1
            state = states[int(deletion_path[i, j])]  # [0]]
            first_seq.append("-")
            second_seq.append(khi[j])
        optimal_path.append(state)
        optimal_path.reverse()
        first_seq.reverse()
        second_seq.reverse()
    return optimal_path, first_seq, second_seq


def em_phmm_alphabeta(l_pairs, alphabet):
    # precision = 0.02
    # diff = 0.3

    # Initialisation
    pseudo_counter = 0.0001
    param = PHMMParameters(0.2, 0.3, 0.4, 0.3, 0.25, pseudo_counter, alphabet)
    # while diff > precision:
    for tour in range(10):
        ksis = []
        gammas = []
        for h in range(len(l_pairs)):
            phi, khi = l_pairs[h]
            alpha, alpha_fin = compute_forward(phi, khi, param.mat_trans.get_transition_matrix(), param.mat_emi_m.mat,
                                               param.mat_emi_is.mat, param.mat_trans.get_final_transition(), alphabet)
            beta = compute_backward(phi, khi, param.mat_trans.get_transition_matrix(), param.mat_emi_m.mat,
                                    param.mat_emi_is.mat, param.mat_trans.get_final_transition(), alphabet)

            transi = param.mat_trans.get_transition_matrix()

            ksis.append(np.zeros((len(phi), len(khi), 3, 3)))
            gammas.append(np.zeros((len(phi), len(khi), 3)))

            for i in range(len(phi)):
                for j in range(len(khi)):
                    for l in States:
                        # calculus of transition probabilities
                        # print(" ksis h, i, j, l, etats : ", alpha[i, j, l.value])
                        # print(" transi h, i, j, l, etats : ", l.value, States.MATCHING.value,
                        # transi[l.value, States.MATCHING.value])
                        # print(" param h, i, j, l, etats : ", param.mat_emi_m.mat[alphabet.index(phi[i]),
                        # alphabet.index(khi[j])])
                        # print(" beta h, i, j, l, etats : ", beta[i, j, States.MATCHING.value])
                        ksis[h][i, j, l.value, States.MATCHING.value] = alpha[i, j, l.value] * \
                                                                        transi[l.value, States.MATCHING.value] * \
                                                                        param.mat_emi_m.mat[alphabet.index(phi[i]),
                                                                                            alphabet.index(khi[j])] * \
                                                                        beta[i, j, States.MATCHING.value]
                        ksis[h][i, j, l.value, States.INSERTION.value] = alpha[i, j, l.value] * \
                                                                         transi[l.value, States.INSERTION.value] * \
                                                                         param.mat_emi_is.mat[alphabet.index(phi[i])] * \
                                                                         beta[i, j, States.INSERTION.value]
                        ksis[h][i, j, l.value, States.DELETION.value] = alpha[i, j, l.value] * \
                                                                        transi[l.value, States.DELETION.value] * \
                                                                        param.mat_emi_is.mat[alphabet.index(khi[j])] * \
                                                                        beta[i, j, States.DELETION.value]
                        # print(alpha_fin)
                        print("alpha", alpha[i, j, l.value], "beta", beta[i, j, :], "transi", transi[l.value, :],
                              "m_emi", param.mat_emi_m.mat[alphabet.index(phi[i]), alphabet.index(khi[j])], "ksi",
                              ksis[h][i, j, l.value, :])
                        ksis[h][i, j, l.value, :] /= np.sum(ksis[h][i, j, l.value, :])  # alpha_fin  # normalisation
                        gammas[h][i, j, l.value] = alpha[i, j, l] * beta[i, j, l.value]
                        # probability of a path given the both sequences
                    gammas[h][i, j, :] /= np.sum(alpha[i, j, :] * beta[i, j, :]) + 0.00000001  # normalisation

        # M

        # estimation de la matrice de transition
        a = np.zeros((3, 3))
        for l in States:
            for m in States:
                a[l.value, m.value] = np.sum(
                    [np.sum(np.sum(ksis[g][:, :, l.value, m.value])) for g in range(len(l_pairs))])
            if np.sum(a[l.value, :]) == 0:
                a[l.value, :] = 0
            else:
                a[l.value, :] /= np.sum(a[l.value, :])
        param.mat_trans.set_freq_transitions(a)
        # TODO problème : on ne peut pas à la fois instancier une matrice entière et instancier par des paramètres
        # param.mat_trans.update_params(1, 1)
        # instancier a dans param de manière correcte !
        # il faut convertir la matrice de transition en paramètres de cette matrice

        # emission in the case of an alignment
        pi_m = np.zeros((len(alphabet), len(alphabet)))
        pi_is = np.zeros((len(alphabet),))

        for carac_1 in range(len(alphabet)):
            for carac_2 in range(len(alphabet)):
                for h in range(len(l_pairs)):
                    phi, khi = l_pairs[h]
                    for i in range(len(phi)):
                        for j in range(len(khi)):
                            pi_m[carac_1, carac_2] = gammas[h][i, j, States.MATCHING.value] * \
                                                     delta(phi[i] == alphabet[carac_1]) * \
                                                     delta(khi[j] == alphabet[carac_2])

        # emission in the cases of insertion or deletion
        for carac in range(len(alphabet)):
            for h in range(len(l_pairs)):
                phi, khi = l_pairs[h]
                for i in range(len(phi)):
                    pi_is[carac] = np.sum(gammas[h][i, :, States.INSERTION.value]) * delta(phi[i] == alphabet[carac])
                for j in range(len(khi)):
                    pi_is[carac] = np.sum(gammas[h][:, j, States.DELETION.value]) * delta(khi[j] == alphabet[carac])

        param.mat_emi_m.set_emission_matrix(pi_m)
        param.mat_emi_is.set_emission_matrix(pi_is)
        diff = param.update_params(1, 1)

    return param


if __name__ == "__main__":
    print(States.MATCHING.value)
    print(States.DELETION.value)
    print(States.INSERTION.value)
    mots = [["eye", "ee", "each", "oog", "oug", "ooch", "a", "auge", "oyg", "øje", "öga", "eyga", "auga",
             "øye", "auga", "augoo"]]
    l_paires = list_to_pairs(mots)
    print(l_paires)
    params = em_phmm_alphabeta(l_paires, alphabet_ger)
    print(params.mat_emi_is.mat)
    print(params.mat_emi_m.mat)
    print(params.mat_trans.get_values())
    mot1 = "oug"
    mot2 = "øye"
    best_alignment, seq1, seq2 = viterbi(mot1, mot2, params)

    print(mot1)
    print(mot2)
    print(seq1)
    print(seq2)
    print(best_alignment)

    # test_list_to_pairs()
