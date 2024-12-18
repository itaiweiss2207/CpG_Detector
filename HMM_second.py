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
def get_second_order_emission_probs(data):
    nucleotides = ["A", "C", "G", "T"]
    NN_counts = Counter()
    NC_counts = Counter()
    CN_counts = Counter()
    CC_counts = Counter()

    for seq, annotation in data:
        for i in range(1, len(annotation) - 1):
            if annotation[i] == "C":
                if annotation[i + 1] == "C":
                    CC_counts[seq[i]] += 1
                else:
                    CN_counts[seq[i]] += 1
            else:
                if annotation[i + 1] == "C":
                    NC_counts[seq[i]] += 1
                else:
                    NN_counts[seq[i]] += 1


    cpg_total = sum(NC_counts.values()) + sum(CC_counts.values())
    non_cpg_total = sum(CN_counts.values()) + sum(NN_counts.values())
    NN_probs = {nuc: NN_counts[nuc] / non_cpg_total for nuc in nucleotides}
    NC_probs = {nuc: NC_counts[nuc] / non_cpg_total for nuc in nucleotides}
    CN_probs = {nuc: CN_counts[nuc] / cpg_total for nuc in nucleotides}
    CC_probs = {nuc: CC_counts[nuc] / cpg_total for nuc in nucleotides}

    return NN_probs, NC_probs, CN_probs, CC_probs


# Second-order transition probabilities
def get_second_order_transition_probs(data, smoothing=1e-6):
    transitions = {
        "CC->CC": 0, "CC->CN": 0, "CN->NC": 0, "CN->NN": 0,
        "NC->CC": 0, "NC->CN": 0, "NN->NC": 0, "NN->NN": 0
    }

    # Count transitions
    for seq, annotation in data:
        for i in range(2, len(annotation)):  # Start from the third position for second-order
            prev_pair = annotation[i - 2] + annotation[i - 1]
            curr_pair = annotation[i - 1] + annotation[i]
            key = f"{prev_pair}->{curr_pair}"
            if key in transitions:
                transitions[key] += 1

    # Add smoothing and normalize
    normalized_transitions = {}
    for key, count in transitions.items():
        prev_state = key.split("->")[0]
        total = sum(value + smoothing for k, value in transitions.items() if k.startswith(prev_state))
        normalized_transitions[key] = (count + smoothing) / total

    return normalized_transitions


# Starting probabilities for second-order HMM
def get_starting_probs(data):
    start_pairs = Counter()
    for seq, annotation in data:
        start_pairs[annotation[:2]] += 1

    total = sum(start_pairs.values())
    return {pair: count / total for pair, count in start_pairs.items()}


# Train a second-degree HMM
def train_second_order_hmm_with_emissions(train_set):
    # Step 1: Get emission probabilities
    NN_probs, NC_probs, CN_probs, CC_probs = get_second_order_emission_probs(train_set)

    # Step 2: Define categorical distributions for each state pair
    NN_distribution = Categorical([[NN_probs["A"], NN_probs["C"], NN_probs["G"], NN_probs["T"]]])
    NC_distribution = Categorical([[NC_probs["A"], NC_probs["C"], NC_probs["G"], NC_probs["T"]]])
    CN_distribution = Categorical([[CN_probs["A"], CN_probs["C"], CN_probs["G"], CN_probs["T"]]])
    CC_distribution = Categorical([[CC_probs["A"], CC_probs["C"], CC_probs["G"], CC_probs["T"]]])

    # Step 3: Transition probabilities (same as first order)
    transition_probs = get_second_order_transition_probs(train_set)  # Should still return {"CC", "CN", "NC", "NN"}
    starting_probs = get_starting_probs(train_set)

    # Step 4: Build the HMM
    model = DenseHMM()

    # Add distributions (augmented states)
    model.add_distributions([CC_distribution, CN_distribution, NC_distribution, NN_distribution])

    # Add edges for starting probabilities
    model.add_edge(model.start, CC_distribution, 0.00001)
    model.add_edge(model.start, CN_distribution, 0.00001)
    model.add_edge(model.start, NC_distribution, 0.00001)
    model.add_edge(model.start, NN_distribution, 1)

    # Add transitions between augmented states
    model.add_edge(CC_distribution, CC_distribution, transition_probs["CC->CC"])
    model.add_edge(CC_distribution, CN_distribution, transition_probs["CC->CN"])
    model.add_edge(CN_distribution, NC_distribution, transition_probs["CN->NC"])
    model.add_edge(CN_distribution, NN_distribution, transition_probs["CN->NN"])
    model.add_edge(NC_distribution, CC_distribution, transition_probs["NC->CC"])
    model.add_edge(NC_distribution, CN_distribution, transition_probs["NC->CN"])
    model.add_edge(NN_distribution, NC_distribution, transition_probs["NN->NC"])
    model.add_edge(NN_distribution, NN_distribution, transition_probs["NN->NN"])

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
    model = train_second_order_hmm_with_emissions(train)
    for seq, annotation in test[0:2]:
        X = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in seq]])
        y_hat = model.predict(X)
        y_hat_string = ''.join([str(y.item()) for y in y_hat[0]])
        y_hat_states = ''.join(["C" if y.item() == 0 else "N" for y in y_hat[0]])
        print("hmm orig: " + annotation)
        print("hmm pred: " + y_hat_states)
        print(hmm_model.loss_over_sequence(annotation, y_hat_states))
    print("Loss, Recall, Precision, F1 over training set:", loss_over_dataset(train, model))
