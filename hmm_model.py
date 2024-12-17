from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from collections import Counter
import numpy as np
import annotate_cpg

training_sequences_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.seq.fa.gz"
training_labels_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.lbl.fa.gz"

# Prepare training data and train model
training_data = annotate_cpg.prepare_training_data(training_sequences_path, training_labels_path)
# remove any pair where the sequence contains a non-nucleotide character
training_data = [pair for pair in training_data if all(char in "AGCT" for char in pair[0])]



# find the emission probabilities for each state
def get_emission_probs(data):
    nucleotides = ["A", "C", "G", "T"]
    cpg_counts = Counter()
    non_cpg_counts = Counter()
    cpg_probs = {}
    non_cpg_probs = {}
    for seq, annotation in data:
        for nucleotide, state in zip(seq, annotation):
            if state == "C":
                cpg_counts[nucleotide] += 1
            elif state == "N":
                non_cpg_counts[nucleotide] += 1
        cpg_total = sum(cpg_counts.values())
        non_cpg_total = sum(non_cpg_counts.values())
        cpg_probs = {nuc: cpg_counts[nuc] / cpg_total for nuc in nucleotides}
        non_cpg_probs = {nuc: non_cpg_counts[nuc] / non_cpg_total for nuc in nucleotides}
    return cpg_probs, non_cpg_probs


def get_transition_probs(data):
    transitions = {"CC": 0, "CN": 0, "NC": 0, "NN": 0}
    for seq, annotation in data:
        for i in range(1, len(annotation)):
            transitions[annotation[i-1] + annotation[i]] += 1
    total_from_C = transitions["CC"] + transitions["CN"]
    total_from_N = transitions["NC"] + transitions["NN"]
    return {"CC": transitions["CC"] / total_from_C, "CN": transitions["CN"] / total_from_C,
            "NC": transitions["NC"] / total_from_N, "NN": transitions["NN"] / total_from_N}


def get_starting_probs(data):
    cpg_starts = 0
    non_cpg_starts = 0
    for seq, annotation in data:
        if annotation[0] == "C":
            cpg_starts += 1
        elif annotation[0] == "N":
            non_cpg_starts += 1
    total_starts = cpg_starts + non_cpg_starts
    return cpg_starts / total_starts, non_cpg_starts / total_starts


def count_differences_numpy(s1, s2):
    min_len = min(len(s1), len(s2))
    arr1 = np.array(list(s1[:min_len]))
    arr2 = np.array(list(s2[:min_len]))
    return np.sum(arr1 != arr2)


def loss_over_sequence(real_states, est_states):
    loss = 0
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for state, state_pred in zip(est_states, real_states):
        if state == "C":
            if state_pred == "C":
                TP += 1
            else:
                FP += 1
                loss += 1
        else:
            if state_pred == "C":
                FN += 1
                loss += 1
            else:
                TN += 1
    loss = loss / len(real_states)
    return loss, TP, FN, FP, TN


def loss_over_dataset(data, model):
    loss, tp, fn, fp, tn = 0, 0, 0, 0, 0
    for seq, annotation in data:
        X = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in seq]])
        pred_states = model.predict(X)
        y_hat_states = ''.join(["C" if y.item() == 0 else "N" for y in pred_states[0]])
        _loss, _tp, _fn, _fp, _tn = loss_over_sequence(annotation, y_hat_states)
        loss += _loss
        tp += _tp
        fn += _fn
        fp += _fp
        tn += _tn
    loss = loss / len(data)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    return loss, recall, precision, F1


def train_model(train_set):
    cpg, non_cpg = get_emission_probs(train_set)
    distribution_cpg = Categorical([[cpg["A"], cpg["G"], cpg["C"], cpg["T"]]])
    distribution_non_cpg = Categorical([[non_cpg["A"], non_cpg["G"], non_cpg["C"], non_cpg["T"]]])
    transition_probs = get_transition_probs(train_set)
    starting_probs = get_starting_probs(train_set)
    print(cpg)
    print(non_cpg)
    print(transition_probs)
    print(starting_probs)

    model = DenseHMM()
    model.add_distributions([distribution_cpg, distribution_non_cpg])

    model.add_edge(model.start, distribution_cpg, starting_probs[0] if starting_probs[0] > 0 else 0.000001)
    model.add_edge(model.start, distribution_non_cpg, starting_probs[1] if starting_probs[1] > 0 else 0.000001)
    model.add_edge(distribution_cpg, distribution_cpg, transition_probs["CC"])
    model.add_edge(distribution_cpg, distribution_non_cpg, transition_probs["CN"])
    model.add_edge(distribution_non_cpg, distribution_cpg, transition_probs["NC"])
    model.add_edge(distribution_non_cpg, distribution_non_cpg, transition_probs["NN"])

    return model

# for seq, annotation in test[0:2]:
#     X = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in seq]])
#     y_hat = model.predict(X)
#     y_hat_string = ''.join([str(y.item()) for y in y_hat[0]])
#     y_hat_states = ''.join(["C" if y.item() == 0 else "N" for y in y_hat[0]])
#     print("hmm orig: " + annotation)
#     print("hmm pred: " + y_hat_states)
#     print(loss_over_sequence(annotation, y_hat_states))


if __name__ == "__main__":

    for proportion in [0.75]:
        # split the data into training and testing sets
        train, test = (training_data[:int(len(training_data) * proportion)],
                       training_data[int(len(training_data) * proportion):])
        model = train_model(train)
        print("proportion: {} \n".format(proportion))
        print("loss, recall, precision, F1 over training set: {} \n".format(loss_over_dataset(train, model)))

# sequence = 'CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC'
# X = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in sequence]])
# print(X.shape)
#
# y_hat = model.predict(X)
#
# print("sequence: {}".format(''.join(sequence)))
# print("hmm pred: {}".format(''.join([str(y.item()) for y in y_hat[0]])))


