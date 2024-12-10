from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from collections import Counter
import numpy as np
import annotate_cpg

training_sequences_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.seq.fa.gz"
training_labels_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.lbl.fa.gz"

# Prepare training data and train model
training_data = annotate_cpg.prepare_training_data(training_sequences_path, training_labels_path)
train, test = training_data[:int(len(training_data) * 0.8)], training_data[int(len(training_data) * 0.8):]


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


cpg, non_cpg = get_emission_probs(train)
distribution_cpg = Categorical([[cpg["A"], cpg["G"], cpg["C"], cpg["T"]]])
distribution_non_cpg = Categorical([[non_cpg["A"], non_cpg["G"], non_cpg["C"], non_cpg["T"]]])
transition_probs = get_transition_probs(train)
print(cpg)
print(non_cpg)
print(transition_probs)

model = DenseHMM()
model.add_distributions([distribution_cpg, distribution_non_cpg])

model.add_edge(model.start, distribution_cpg, 0.5)
model.add_edge(model.start, distribution_non_cpg, 0.5)
model.add_edge(distribution_cpg, distribution_cpg, transition_probs["CC"])
model.add_edge(distribution_cpg, distribution_non_cpg, transition_probs["CN"])
model.add_edge(distribution_non_cpg, distribution_cpg, transition_probs["NC"])
model.add_edge(distribution_non_cpg, distribution_non_cpg, transition_probs["NN"])


sequence = 'CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC'
X = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in sequence]])
print(X.shape)

y_hat = model.predict(X)

print("sequence: {}".format(''.join(sequence)))
print("hmm pred: {}".format(''.join([str(y.item()) for y in y_hat[0]])))

