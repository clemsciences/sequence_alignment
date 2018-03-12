# -*-coding:utf-8-*-

import numpy as np
import numpy.random as nr
from setuptools.command.test import test

from utils import *

__author__ = "Clément Besnier <clemsciences@aol.com>"


alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


class CoupleEmissionMatrix:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.n = len(alphabet)
        self.mat = np.zeros((self.n, self.n))

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
        self.mat = np.zeros((self.n,))

    def get_emission_matrix(self):
        return self.mat

    def set_emission_matrix(self, mat):
        self.mat = mat

    def __getitem__(self, item):
        return self.mat

    def __len__(self):
        return self.n


class EmissionMatrix:
    def __init__(self, epsilon, delta, tau_m, tau_is, _lambda):
        self.epsilon = epsilon
        self.delta = delta
        self.tau_m = tau_m
        self.tau_is = tau_is
        self._lambda = _lambda
        self.freq_transitions = np.zeros((3, 3))
        # def add_transition(self, l, m):

    def get_values(self):
        return self.epsilon, self.delta, self.tau_m, self.tau_is, self._lambda

    def update_epsilon(self):
        nume = self.freq_transitions[States.INSERTION, States.INSERTION] + self.freq_transitions[States.DELETION, States.DELETION]
        denomi = np.sum(self.freq_transitions[States.INSERTION, :]) + np.sum(self.freq_transitions[States.DELETION, :])
        self.epsilon = nume / denomi

    def update_delta(self):
        nume = self.freq_transitions[States.MATCHING, States.INSERTION] + self.freq_transitions[States.CORRESPONDANCE, States.DELETION]
        denomi = np.sum(self.freq_transitions[States.CORRESPONDANCE, :])
        self.delta = nume /denomi

    def update_tau_m(self, n_m_fin):
        nume = n_m_fin
        denomi = n_m_fin + np.sum(self.freq_transitions[States.CORRESPONDANCE, :])
        self.tau_m = nume / denomi

    def update_tau_is(self, n_is_fin):
        nume = n_is_fin
        denomi = n_is_fin + np.sum(self.freq_transitions[States.INSERTION, :]) + np.sum(self.freq_transitions[States.DELETION, :])
        self.tau_is = nume / denomi

    def update_lambda(self):
        nume = self.freq_transitions[States.E_INSER, States.DELETION] + self.freq_transitions[States.DELETION, States.INSERTION]
        denomi = np.sum(self.freq_transitions[States.INSERTION, :]) + np.sum(self.freq_transitions[States.DELETION, :])
        self._lambda = nume / denomi

    def get_transition_matrix(self):
        return np.array([[1-2*self.delta, self.delta, self.delta],
                         [1-self.epsilon-self._lambda, self.epsilon, self._lambda],
                         [1-self._lambda-self.epsilon, self._lambda, self.epsilon]])

    def get_final_transition(self):
        return np.array([self.tau_m, self.tau_is, self.tau_is])

    def update_params(self, n_m_fin, n_is_fin):
        self.update_epsilon()
        self.update_delta()
        self.update_tau_m(n_m_fin)
        self.update_tau_is(n_is_fin)
        self.update_lambda()


class PHMMParameters:
    def __init__(self, epsilon, delta, tau_m, tau_is, _lambda, pseudo_compte):
        self.mat_trans = EmissionMatrix(epsilon, delta, tau_m, tau_is, _lambda)
        self.mat_emi_m = CoupleEmissionMatrix(alphabet)
        self.mat_emi_is = SimpleEmissionMatrix(alphabet)
        self.pseudo_compte = pseudo_compte

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
        epsilon, delta, tau_m, tau_is, _lambda = self.mat_trans.get_values()
        self.mat_trans.update_params(n_m_fin, n_is_fin)
        diff = abs(epsilon- self.mat_trans.epsilon) + abs(delta - self.mat_trans.delta) + abs(tau_m - self.mat_trans.tau_m) +\
               abs(tau_is - self.mat_trans.tau_is) + abs(_lambda - self.mat_trans._lambda)
        return diff


def compute_forward(phi, khi, mat_tra, mat_emi_m, mat_emi_is, trans_fin):
    alpha = np.zeros((phi.size+1, khi.size+1, mat_tra.shape[0]))
    alpha[0, 0, 0] = 1
    for i in range(phi.size):
        for j in range(khi.size):
            if i == 0 and j == 0:
                continue
            if i == 0:
                alpha[i, j, 0] = 0
                alpha[i, j, 1] = 0
                alpha[i, j, 2] = mat_emi_is[alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[2, :] * alpha[i, j-1, :])
            if j == 0:
                alpha[i, j, 0] = 0
                alpha[i, j, 1] = mat_emi_is[alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[2, :] * alpha[i-1, j, :])
                alpha[i, j, 2] = 0

            else:
                alpha[i, j, 0] = mat_emi_m[alphabet.index(phi[i]), alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[0, :] * alpha[i-1, j-1, :])

                alpha[i, j, 1] = mat_emi_is[alphabet.index(phi[i])] * \
                                 np.sum(mat_tra[1, :] * alpha[i-1, j, :])
                alpha[i, j, 2] = mat_emi_is[alphabet.index(khi[j])] * \
                                 np.sum(mat_tra[2, :] * alpha[i, j-1, :])
                alpha[i, j, :] /= np.sum(alpha[i, j, :])
    alpha_fin = np.sum(trans_fin * alpha[phi.size, khi.size, :])
    return alpha, alpha_fin


def compute_backward(phi, khi, mat_tra, mat_emi_m, mat_emi_is, trans_fin):
    beta = np.zeros((phi.size+1, khi.size+1, mat_tra.shape[0]))
    beta[phi.size, khi.size, :] = trans_fin
    for i in range(phi.size, -1, -1):
        for j in range(khi.size, -1, -1):
            for k in States:
                print(mat_emi_m.shape)
                print(mat_emi_is.shape)
                print(mat_emi_m)
                emi = np.array([mat_emi_m[alphabet.index(phi[i+1]), alphabet.index(khi[j+1])],
                                                 mat_emi_is[alphabet.index(phi[i+1])], mat_emi_is[alphabet.index(khi[j+1])]])
                beta_transition = np.array([beta[i+1, j+1, 0], beta[i+1, j, 1], beta[i, j+1, 2]])
                beta[i, j, k] = np.sum((emi * mat_tra[k, :]) * beta_transition)
    return beta


def viterbi(phi, khi, param: PHMMParameters):
    n = len(phi)
    m = len(khi)

    matches = np.zeros((n+1, m+1))
    matching_path = np.zeros((n+1, m+1))
    insertions = np.zeros((n+1, m+1))
    insertion_path = np.zeros((n+1, m+1))
    deletions = np.zeros((n+1, m+1))
    deletion_path = np.zeros((n+1, m+1))

    matches[0, 0] = 1
    insertions[:, 0] = 0
    insertions[0, :] = 0
    deletions[:, 0] = 0
    deletions[0, :] = 0
    optimal_path = []

    for i in range(1, n+1):
        for j in range(1, m+1):
            matches[i, j] = param.mat_emi_m.mat[alphabet.index(phi[i]), alphabet.index(khi[j])]*np.max([(1-2*param.mat_trans.delta- param.mat_trans.tau_m)* matches[i-1, j-1],
                                                                                                        (1-param.mat_trans.epsilon-param.mat_trans.tau_is - param.mat_trans._lambda) * insertions[i-1, j-1],
                                                                         (1-param.mat_trans.epsilon-param.mat_trans.tau_is - param.mat_trans._lambda) * deletions[i-1, j-1]])
            matching_path[i, j] = np.argmax([(1-2*param.mat_trans.delta- param.mat_trans.tau_m)* matches[i-1, j-1],
                                                                        (1-param.mat_trans.epsilon-param.mat_trans.tau_is - param.mat_trans._lambda) * insertions[i-1, j-1],
                                                                         (1-param.mat_trans.epsilon-param.mat_trans.tau_is - param.mat_trans._lambda) * deletions[i-1, j-1]])

            insertions[i, j] = param.mat_emi_is[alphabet.index(khi[j])]*np.max([param.mat_trans.delta* matches[i-1, j],
                                                                        param.mat_trans.epsilon * insertions[i-1, j]])
            insertion_path[i, j] = np.argmax([param.mat_trans.delta* matches[i-1, j], param.mat_trans.epsilon * insertions[i-1, j]])
            deletions[i, j] = param.mat_emi_is[alphabet.index(khi[j])] * np.max([param.mat_trans.delta* matches[i, j-1],
                                                                        param.mat_trans.epsilon * deletions[i, j-1]])
            deletion_path[i, j] = np.argmax([param.mat_trans.delta* matches[i, j-1], param.mat_trans.epsilon * deletions[i, j-1]])

            if i == n and j == m:
                end = np.max([param.mat_trans.tau_m * matches[n, m], param.mat_trans.tau_is*insertions[n, m], param.mat_trans.tau_is*deletions[n, m]])
                final_state = np.argmax([param.mat_trans.tau_m * matches[n, m], param.mat_trans.tau_is*insertions[n, m], param.mat_trans.tau_is*deletions[n, m]])

    i = n
    j = m
    state = states[final_state[0]]
    optimal_path.append(state)
    while i > 0 and j > 0:
        if state == "M":
            i -= 1
            j -= 1
            state = states[matching_path[i, j][0]]
        elif state == "I":
            i -= 1
            state = states[insertion_path[i, j][0]]
        elif state == "S":
            j -= 1
            state = states[deletion_path[i, j][0]]
        optimal_path.append(state)

    return optimal_path


def em_phmm_alphabeta(l_paires):
    precision = 0.02
    diff = 0.3

    # Initialisation
    pseudo_compte = 0.0001
    param = PHMMParameters(nr.random(), nr.random(), nr.random(), nr.random(), nr.random(), pseudo_compte)

    while diff > precision:
        ksis = []
        gammas = []
        for h in range(len(l_paires)):
            phi, khi = l_paires[h]
            alpha, alpha_fin = compute_forward(phi, khi, param.mat_trans.get_transition_matrix(), param.mat_emi_m.mat,
                                    param.mat_emi_is.mat, param.mat_trans.get_final_transition())
            beta = compute_backward(phi, khi, param.mat_trans.get_transition_matrix(), param.mat_emi_m.mat,
                                    param.mat_emi_is.mat , param.mat_trans.get_final_transition())

            transi = param.mat_trans.get_transition_matrix()

            ksis.append(np.zeros((len(phi), len(khi), 3, 3)))
            gammas.append(np.zeros((len(phi), len(khi), 3)))

            for i in range(len(phi)):
                for j in range(len(khi)):
                    for l in States:
                        # calculus of transition probabilities
                        ksis[h][i, j, l, States.CORRESPONDANCE] = alpha[i, j, l] * transi[l, States.CORRESPONDANCE] * param.mat_emi_m[phi[i], khi[j]] * beta[i, j, States.INSERTION]
                        ksis[h][i, j, l, States.INSERTION] = alpha[i, j, l] * transi[l, States.INSERTION] * param.mat_emi_is[phi[i]] * beta[i, j, States.INSERTION]
                        ksis[h][i, j, l, States.DELETION] = alpha[i, j, l] * transi[l, States.DELETION] * param.mat_emi_is[khi[i]] * beta[i, j, States.DELETION]
                        ksis[h][i, j, l, :] /= alpha_fin  # normalisation
                        gammas[h][i, j, l] = alpha[i, j, l]*beta[i, j, l]
                        # probability of a path given the both sequences
                    gammas[h][i, j, :] /= np.sum(alpha[i, j, :]*beta[i, j, :])  # normalisation

        # M

        # estimation de la matrice de transition
        a = np.zeros((3, 3))
        for l in States:
            for m in States:
                a[l, m] = np.sum([np.sum(np.sum(ksis[h][:, :, l, m])) for h in range(len(l_paires))])
            a[l, :] /= np.sum(a[l, :])
        # instancier a dans param de manière correcte !
        # il faut convertir la matrice de transition en paramètres de cette matrice

        # emission in the case of an alignment
        pi_m = np.zeros((len(alphabet), len(alphabet)))
        pi_is = np.zeros((2, len(alphabet)))
        for carac_1 in range(len(alphabet)):
            for carac_2 in range(len(alphabet)):
                for h in range(len(l_paires)):
                    phi, khi = l_paires[h]
                    for i in range(len(phi)):
                        for j in range(len(khi)):
                            pi_m[carac_1, carac_2] = gammas[h][i, j, l] * delta(phi[i] == alphabet[carac_1]) * delta(khi[j] == alphabet[carac_2])

        # emission in the cases of insertion or deletion
        for carac in range(len(alphabet)):
            for h in range(len(l_paires)):
                phi, khi = l_paires[h]
                for i in range(len(phi)):
                    pi_is[States.INSERTION, carac] = np.sum(gammas[h][i, :, l]) * delta(phi[i] == alphabet[carac])
                for j in range(len(khi)):
                    pi_is[States.DELETION, carac] = np.sum(gammas[h][:, j, l]) * delta(khi[j] == alphabet[carac])

        param.mat_emi_m.set_emission_matrix(pi_m)
        param.mat_emi_m.set_emission_matrix(pi_is)
        # diff = param.update_params(n_m_fin, n_is_fin)

    return param


if __name__ == "__main__":
    # l_paires = []
    # param, freq_emissions = em_phmm(l_paires)
    # mot1 = ""
    # mot2 = ""
    # meilleur_alignement = viterbi(mot1, mot2, param, freq_emissions)
    # print(mot1)
    # print(mot2)
    # print(meilleur_alignement)

    test_list_to_pairs()