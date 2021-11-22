import numpy as np
from utils import  *
from viterbi import *

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

def count_transitions_and_emissions(K, D, x, z):
    length = len(x)
    transition_count = np.ones(shape=(K, K))
    emission_count = np.ones(shape=(K, D))
    print("length = ", length) #2211457
    for i in range(0, length-1):
        transition_count[z[i]][z[i+1]] += 1
    transition_row_sum = np.sum(transition_count, axis=1).reshape(K, 1)
    transition_count = transition_count/transition_row_sum

    for i in range(0, length):
        emission_count[z[i]][x[i]] += 1
    emission_row_sum = np.sum(emission_count, axis=1).reshape(K, 1)
    emission_count = emission_count/emission_row_sum

    return transition_count, emission_count

def training_by_counting(K, D, x, z):
    initial_count = np.ones(shape=(K, 1))
    initial_count[z[0]][0] += 1
    initial_sum = np.sum(initial_count)
    initial_count = initial_count/initial_sum
    transition_count, emission_count = count_transitions_and_emissions(K, D,  x, z)
    Hmm = hmm(initial_count, transition_count, emission_count)
    return Hmm
