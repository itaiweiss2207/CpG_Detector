import argparse
import gzip
from Bio import SeqIO
import numpy as np
from hmmlearn import hmm
from hmmlearn.base import _BaseHMM
from matplotlib import pyplot as plt
from sklearn.metrics import auc

import annotate_cpg
import hmm_model


def get_emission_probs(data):
    # Define nucleotide indices for easy lookup
    nucleotides = "ACGT"
    states = "NC"
    n = len(nucleotides)

    # Initialize count matrices for each state
    counts = {state: np.zeros((n, n), dtype=np.float64) for state in states}

    # Iterate over the data
    for seq, annotation in data:
        for i in range(1, len(seq)):  # Start from the second nucleotide
            prev_nucleotide = seq[i - 1]
            curr_nucleotide = seq[i]
            curr_state = annotation[i]

            # Map nucleotides and states to indices
            prev_idx = nucleotides.index(prev_nucleotide)
            curr_idx = nucleotides.index(curr_nucleotide)

            # Update the count for the current state
            counts[curr_state][prev_idx, curr_idx] += 1
    # print("Counts: " + str(counts))
    # Convert counts to probabilities by normalizing
    emission_probs = {}
    for state in states:
        total_counts = counts[state].sum(axis=1, keepdims=True)
        # Avoid division by zero by replacing zeros with ones (will lead to zero probabilities)
        total_counts[total_counts == 0] = 1
        emission_probs[state] = counts[state] / total_counts

    # return probs as array of shape (2, 4, 4)
    emission_probs = np.array([emission_probs[state] for state in states])
    return emission_probs

def get_transition_probs(data):
    transitions = {"CC": 0, "CN": 0, "NC": 0, "NN": 0}
    for seq, annotation in data:
        for i in range(1, len(annotation)):
            transitions[annotation[i - 1] + annotation[i]] += 1
    total_from_C = transitions["CC"] + transitions["CN"]
    total_from_N = transitions["NC"] + transitions["NN"]
    dict =  {"CC": transitions["CC"] / total_from_C, "CN": transitions["CN"] / total_from_C,
            "NC": transitions["NC"] / total_from_N, "NN": transitions["NN"] / total_from_N}
    matrix = np.array([[dict["NN"], dict["NC"]], [dict["CN"], dict["CC"]]])
    return matrix

# Mapping for nucleotide bases to integers
nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def encode_sequence(sequence):
    """
    Encode a DNA sequence as a list of integers, where each nucleotide
    is mapped to an integer (A=0, C=1, G=2, T=3).
    """
    # Ensure that the sequence is processed one nucleotide at a time
    return [[nucleotide_map[nucleotide]] for nucleotide in sequence if nucleotide in nucleotide_map]


class ConditionalHMM(_BaseHMM):

    def _init_(self, n_components, n_iter=10000, tol=1e-5, random_state=None):
        super()._init_(n_components=n_components, n_iter=n_iter, tol=tol, random_state=random_state)
        self.emission_matrix = None  # Shape: (n_components, 4, 4)

    def _initialize_emission_matrix(self, emission_probs=None):
        """
        Initialize the emission matrix (n_states x 4 x 4).
        """
        n_states = self.n_components
        self.emission_matrix = None

    def _compute_log_likelihood(self, X):
        """
        Compute log likelihood of observations given the state and the previous observation.
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        log_prob : array, shape (n_samples, n_components)
            Emission log probability of each sample in `X` for each of the
            model states, i.e., `log(p(X|state))`.
        """
        X = np.array(X)  # Ensure X is a NumPy array

        log_likelihood = np.zeros((X.shape[0], self.n_components))

        # Compute log-likelihood for each observation
        for t in range(1, X.shape[0]):
            prev_obs = X[t - 1, 0]
            curr_obs = X[t, 0]
            log_likelihood[t] = np.log(self.emission_matrix[:, prev_obs, curr_obs])
        return log_likelihood

    def set_emission_matrix(self, emission_matrix):
        """
        Set the custom emission matrix manually.
        """
        if emission_matrix.shape != (self.n_components, 4, 4):
            raise ValueError("Emission matrix must be of shape (n_components, 4, 4)")
        self.emission_matrix = emission_matrix


# Integration: Adjusting the train_model function
def train_model(train_data):
    # Get probabilities
    emission_probs = get_emission_probs(train_data)
    # print(emission_probs)
    transition_probs = get_transition_probs(train_data)
    # print(transition_probs)
    starting_probs = [1.0, 0.0]

    # Initialize the model
    model = ConditionalHMM(n_components=2)
    model.set_emission_matrix(emission_probs)
    model.startprob_ = np.array(starting_probs)
    model.transmat_ = transition_probs

    return model


def loss_over_dataset(data, model):
    loss, tp, fn, fp, tn = 0, 0, 0, 0, 0
    for seq, annotation in data:
        encoded_seq = encode_sequence(seq)
        pred_states = model.predict(encoded_seq)
        y_hat_states = ''.join(["C" if y == 1 else "N" for y in pred_states])
        _loss, _tp, _fn, _fp, _tn = hmm_model.loss_over_sequence(annotation, y_hat_states)
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


def train_ensemble(train_data, num_models, pct_data_per_model=0.8):
    models = []
    for i in range(num_models):
        # random subset of the data (seq and annotations)
        data = [train_data[i] for i in np.random.choice(len(train_data), int(len(train_data) * pct_data_per_model))]
        models.append(train_model(data))
    return models


def predict_from_ensemble(seq, annotation, models, alpha=0.5):
    pred_states = []
    for model in models:
        encoded_seq = encode_sequence(seq)
        pred_states.append(model.predict(encoded_seq))
    pred_states = np.array(pred_states)
    pred_states = np.mean(pred_states, axis=0)
    y_hat_states = ''.join(["C" if y > alpha else "N" for y in pred_states])
    return hmm_model.loss_over_sequence(annotation, y_hat_states)

def loss_dataset_from_ensemble(data, models, alpha=0.5):
    loss, tp, fn, fp, tn = 0, 0, 0, 0, 0
    for seq, annotation in data:
        _loss, _tp, _fn, _fp, _tn = predict_from_ensemble(seq, annotation, models, alpha)
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

def compute_roc_curve_single_model(data, model):
    alphas = np.linspace(0, 1, 101)  # Thresholds from 0.0 to 1.0
    tpr_list, fpr_list = [], []

    for alpha in alphas:
        tp, fn, fp, tn = 0, 0, 0, 0

        for seq, annotation in data:
            # Encode sequence
            encoded_seq = encode_sequence(seq)

            # Get posterior probabilities for each state (shape: [len(seq), n_states])
            log_prob = model._compute_log_likelihood(encoded_seq)
            state_probs = np.exp(log_prob)  # Convert log probabilities to probabilities

            # Get "C" state probabilities
            c_probs = state_probs[:, 1]  # Column for the CpG ("C") state

            # Threshold to classify each position
            pred_states = ["C" if p > alpha else "N" for p in c_probs]

            # Compare predictions to true annotations
            for i in range(len(annotation)):
                if annotation[i] == "C" and pred_states[i] == "C":
                    tp += 1
                elif annotation[i] == "C" and pred_states[i] == "N":
                    fn += 1
                elif annotation[i] == "N" and pred_states[i] == "C":
                    fp += 1
                elif annotation[i] == "N" and pred_states[i] == "N":
                    tn += 1

        # Calculate TPR and FPR for the current threshold
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Compute Area Under the Curve (AUC)
    roc_auc = auc(fpr_list, tpr_list)

    # Plot the ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'r--', label="Random Guessing (AUC = 0.5)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve for Single Model")
    plt.legend(loc="lower right")
    plt.show()

    return tpr_list, fpr_list, roc_auc


if __name__ == "__main__":

    # Paths
    training_sequences_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.seq.fa.gz"
    training_labels_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.lbl.fa.gz"

    all_data = annotate_cpg.prepare_training_data(training_sequences_path, training_labels_path)
    all_data = [pair for pair in all_data if all(char in "ACGT" for char in pair[0])]

    # Check data validity
    for seq, annotation in all_data:
        if len(seq) != len(annotation):
            print(f"Sequence and annotation length mismatch: {len(seq)} vs {len(annotation)}")
        invalid_chars = [char for char in seq if char not in 'ACGT']
        if invalid_chars:
            print(f"Invalid characters in sequence: {invalid_chars}")

    for proportion in [0.75]:
        train = all_data[:int(len(all_data) * proportion)]
        test = all_data[int(len(all_data) * proportion):]
        model = train_model(train)

        print("Proportion: {} \n".format(proportion))
        print("Loss, recall, precision, F1 over training set: {} \n".format(
            loss_over_dataset(train, model)
        ))
        print("Loss, recall, precision, F1 over test set: {} \n".format(
            loss_over_dataset(test, model)
        ))

        # Compute ROC curve for the single model
        _, _, auc = compute_roc_curve_single_model(test, model)
        print(auc)


    #
    # train_data = all_data[:int(len(all_data) * 0.75)]
    # ensemble = train_ensemble(train_data, 20)
    # print(loss_dataset_from_ensemble(all_data, ensemble, alpha=0.7))



