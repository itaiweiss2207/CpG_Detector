from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from collections import Counter
import numpy as np
import annotate_cpg
import hmm_model

# Paths to data
training_sequences_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.seq.fa.gz"
training_labels_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.lbl.fa.gz"

# Prepare and clean the data
training_data = annotate_cpg.prepare_training_data(training_sequences_path, training_labels_path)
training_data = [pair for pair in training_data if all(char in "AGCT" for char in pair[0])]


# Find emission probabilities
def get_emission_probs(data):
    nucleotides = ["A", "C", "G", "T"]
    cpg_counts = Counter()
    non_cpg_counts = Counter()

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


# Second-order transition probabilities
def get_second_order_transition_probs(data):
    transitions = Counter()

    for seq, annotation in data:
        for i in range(2, len(annotation)):
            prev_pair = annotation[i - 2] + annotation[i - 1]
            current_state = annotation[i]
            key = (prev_pair, current_state)
            transitions[key] += 1

    # Normalize transitions
    transition_probs = {}
    for (prev_pair, current_state), count in transitions.items():
        total = sum(transitions[(prev_pair, s)] for s in "CN")
        transition_probs[(prev_pair, current_state)] = count / total

    return transition_probs


# Starting probabilities for second-order HMM
def get_starting_probs(data):
    start_pairs = Counter()
    for seq, annotation in data:
        start_pairs[annotation[:2]] += 1

    total = sum(start_pairs.values())
    return {pair: count / total for pair, count in start_pairs.items()}


# Train a second-degree HMM
def train_second_order_hmm(train_set):
    cpg_probs, non_cpg_probs = get_emission_probs(train_set)
    transition_probs = get_second_order_transition_probs(train_set)
    starting_probs = get_starting_probs(train_set)

    print("Emission probabilities (CpG):", cpg_probs)
    print("Emission probabilities (Non-CpG):", non_cpg_probs)
    print("Starting probabilities:", starting_probs)
    print("Transition probabilities:", transition_probs)

    # Emission distributions
    distribution_cpg = Categorical([[cpg_probs["A"], cpg_probs["C"], cpg_probs["G"], cpg_probs["T"]]])
    distribution_non_cpg = Categorical([[non_cpg_probs["A"], non_cpg_probs["C"], non_cpg_probs["G"], non_cpg_probs["T"]]])

    # Build model
    model = DenseHMM()
    model.add_distributions([distribution_cpg, distribution_non_cpg])

    # Add second-order transitions
    for prev_pair in ["CC", "CN", "NC", "NN"]:
        for current_state, distribution in zip("CN", [distribution_cpg, distribution_non_cpg]):
            prob = transition_probs.get((prev_pair, current_state), 0.000001)  # Small prob if missing
            model.add_edge(prev_pair, distribution, prob)

    # Starting edges for state pairs
    for pair, prob in starting_probs.items():
        model.add_edge(model.start, pair, prob)

    return model


# Loss calculation remains unchanged
def loss_over_dataset(data, model):
    loss, tp, fn, fp, tn = 0, 0, 0, 0, 0
    for seq, annotation in data:
        X = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in seq]])
        pred_states = model.predict(X)
        y_hat_states = ''.join(["C" if y.item() == 0 else "N" for y in pred_states[0]])
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


if __name__ == "__main__":
    proportion = 0.75
    train, test = training_data[:int(len(training_data) * proportion)], training_data[int(len(training_data) * proportion):]
    model = train_second_order_hmm(train)
    print("Loss, Recall, Precision, F1 over training set:", loss_over_dataset(train, model))
