import numpy as np
from hmmlearn.base import _BaseHMM
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def get_emission_probs(data):
    # Define nucleotide indices for easy lookup
    prev_combinations = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    nucleotides = "ACGT"
    states = "NC"

    # Initialize count matrices for each state
    counts = {state: np.zeros((len(prev_combinations), len(nucleotides)), dtype=np.float64) for state in states}

    # Iterate over the data
    for seq, annotation in data:
        for i in range(2, len(seq)):  # Start from the second nucleotide
            prev_comb = seq[i - 2:i]
            curr_nucleotide = seq[i]
            curr_state = annotation[i]

            # Map nucleotides and states to indices
            prev_idx = prev_combinations.index(prev_comb)
            curr_idx = nucleotides.index(curr_nucleotide)

            # Update the count for the current state
            counts[curr_state][prev_idx, curr_idx] += 1
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
    print(matrix.shape)
    return matrix

def encode_sequence(sequence):
    """
    Encode a DNA sequence as a list of integers, where each nucleotide
    is mapped to an integer (A=0, C=1, G=2, T=3).
    """

    # Mapping for nucleotide bases to integers
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Ensure that the sequence is processed one nucleotide at a time
    return [[nucleotide_map[nucleotide]] for nucleotide in sequence if nucleotide in nucleotide_map]


class ConditionalHMM(_BaseHMM):

    def _init_(self, n_components, n_iter, tol, random_state=None):
        super()._init_(n_components=n_components, n_iter=n_iter, tol=tol, random_state=random_state)
        self.emission_matrix = None  # Shape: (n_components, 4, 4)
        self.threshold = 0.5

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
        log_likelihood = np.zeros((X.shape[0], self.n_components))
        prev_combinations = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA",
                             "TC", "TG", "TT"]
        prev_combinations = [''.join(str(nucleotide[0]) for nucleotide in encode_sequence(comb)) for comb in
             prev_combinations]
        # Compute log-likelihood for each observation
        for t in range(2, X.shape[0]):
            prev_comb = str(X[t - 2,0]) + str(X[t - 1,0])
            prev_obs = prev_combinations.index(prev_comb)
            curr_obs = X[t, 0]
            log_likelihood[t] = np.log(self.emission_matrix[:, prev_obs, curr_obs])
        return log_likelihood

    def set_emission_matrix(self, emission_matrix):
        """
        Set the custom emission matrix manually.
        """
        if emission_matrix.shape != (self.n_components, 16, 4):
            raise ValueError("Emission matrix must be of shape (n_components, 16, 4)")
        self.emission_matrix = emission_matrix

    def predict_with_threshold(self, seq):
        encoded_seq = encode_sequence(seq)
        state_probs = self.predict_proba(encoded_seq)
        return ''.join(["C" if prob > self.threshold else "N" for prob in state_probs[:, 1]])

    def find_best_threshold(self, dataset, proportion=0.75):
        train = dataset[:int(len(dataset) * proportion)]
        test = dataset[int(len(dataset) * proportion):]
        model = train_model(train)
        best_thresh = build_roc_curve(test, model)
        self.threshold = best_thresh



# Integration: Adjusting the train_model function
def train_model(train_data):
    # Get probabilities
    emission_probs = get_emission_probs(train_data)
    print(emission_probs)
    transition_probs = get_transition_probs(train_data)
    print(transition_probs)
    starting_probs = [1.0, 0.0]

    # Initialize the model
    model = ConditionalHMM(n_components=2)
    model.set_emission_matrix(emission_probs)
    model.startprob_ = np.array(starting_probs)
    model.transmat_ = transition_probs

    return model

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
    count = 0
    for seq, annotation in data:
        encoded_seq = encode_sequence(seq)
        pred_states = model.predict(encoded_seq)
        y_hat_states = ''.join(["C" if y == 1 else "N" for y in pred_states])
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
    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
    return loss, recall, precision, F1, balanced_accuracy


def build_roc_curve(data, model):
    y_true = []
    y_pred_prob = []

    for seq, annotation in data:
        encoded_seq = encode_sequence(seq)
        state_probs = model.predict_proba(encoded_seq)
        y_pred_prob.extend(state_probs[:, 1])
        y_true.extend([1 if char == "C" else 0 for char in annotation])

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    f1 = 2 * tpr * (1 - fpr) / (tpr + 1 - fpr)

    num_positive_annotations = sum(1 for char in y_true if char == 1)
    num_negative_annotations = sum(1 for char in y_true if char == 0)
    weighted_accuracy = (num_positive_annotations * tpr + num_negative_annotations * (1 - fpr)) / (num_positive_annotations + num_negative_annotations)

    print("Best F1: ", max(f1))
    print("Best weighted accuracy: ", max(weighted_accuracy))
    # plot curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, AUC = ' + str(roc_auc_score(y_true, y_pred_prob)))
    plt.show()

    # return the threshold for the best weighted accuracy
    return thresholds[np.argmax(weighted_accuracy)]


if __name__ == "__main__":

    # Paths
    # training_sequences_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.seq.fa.gz"
    # training_labels_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.lbl.fa.gz"
    #
    # all_data = annotate_cpg.prepare_training_data(training_sequences_path, training_labels_path)
    # all_data = [pair for pair in all_data if all(char in "ACGT" for char in pair[0])]
    #
    # # Check data validity
    # for seq, annotation in all_data:
    #     if len(seq) != len(annotation):
    #         print(f"Sequence and annotation length mismatch: {len(seq)} vs {len(annotation)}")
    #     invalid_chars = [char for char in seq if char not in 'ACGT']
    #     if invalid_chars:
    #         print(f"Invalid characters in sequence: {invalid_chars}")
    #
    # best_thresh = 0
    # for proportion in [0.75]:
    #     train = all_data[:int(len(all_data) * proportion)]
    #     test = all_data[int(len(all_data) * proportion):]
    #     model = train_model(train)
    #
    #     print("Proportion: {} \n".format(proportion))
    #     print("Loss, recall, precision, F1, b_acc over test set: {} \n".format(
    #         loss_over_dataset(test, model)
    #     ))
    #     best_thresh = build_roc_curve(test, model)
    pass

