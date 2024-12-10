from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from collections import Counter
import numpy as np

#
# # Sample training data: List of tuples (sequence, annotation)
# training_data = [
#     ("ACGTACGTACGT", "NNNNCCCCNNNN"),
#     ("CGTACGTACGTA", "NNCCCCCNNNNN"),
#     # Add more sequences and annotations here
# ]
#
# # Step 1: Calculate emission probabilities for each state (CpG and non-CpG)
# def calculate_emission_probabilities(training_data):
#     nucleotides = ["A", "C", "G", "T"]
#     cpg_counts = Counter()
#     non_cpg_counts = Counter()
#
#     for sequence, annotation in training_data:
#         for nucleotide, state in zip(sequence, annotation):
#             if state == "C":  # CpG island
#                 cpg_counts[nucleotide] += 1
#             elif state == "N":  # Non-CpG island
#                 non_cpg_counts[nucleotide] += 1
#
#     # Normalize counts to probabilities
#     cpg_total = sum(cpg_counts.values())
#     non_cpg_total = sum(non_cpg_counts.values())
#
#     cpg_probs = {nuc: cpg_counts[nuc] / cpg_total for nuc in nucleotides}
#     non_cpg_probs = {nuc: non_cpg_counts[nuc] / non_cpg_total for nuc in nucleotides}
#
#     return cpg_probs, non_cpg_probs
#
# cpg_probs, non_cpg_probs = calculate_emission_probabilities(training_data)
#
# # Step 2: Define states using the emission probabilities
# cpg_distribution = Categorical(cpg_probs)
# non_cpg_distribution = Categorical(non_cpg_probs)
#
# # Step 3: Create and train the HMM
# hmm = DenseHMM()
# hmm.add_distributions([cpg_distribution, non_cpg_distribution])
#
# # Define transition probabilities (adjust based on your data)
# hmm.add_edge(hmm.start, cpg_distribution, 0.5)
# hmm.add_edge(hmm.start, non_cpg_distribution, 0.5)
# hmm.add_edge(cpg_distribution, cpg_distribution, 0.9)
# hmm.add_edge(cpg_distribution, non_cpg_distribution, 0.1)
# hmm.add_edge(non_cpg_distribution, non_cpg_distribution, 0.9)
# hmm.add_edge(non_cpg_distribution, cpg_distribution, 0.1)
#
# # Finalize the model
# # hmm.bake()
#
# # Step 4: Train the HMM (optional, if you want to optimize transitions)
# # Convert sequences to observation format
# training_sequences = [list(sequence) for sequence, _ in training_data]
# hmm.fit(training_sequences, algorithm="baum-welch")
#
# # Step 5: Test the HMM
# test_sequence = list("ACGTACGTACGT")
# log_probability, state_path = hmm.viterbi(test_sequence)
#
# print("Log Probability:", log_probability)
# print("State Path:", " ".join(state.name for state in state_path))


d1 = Categorical([[0.25, 0.25, 0.25, 0.25]])
d2 = Categorical([[0.10, 0.40, 0.40, 0.10]])

model = DenseHMM()
model.add_distributions([d1, d2])

model.add_edge(model.start, d1, 0.5)
model.add_edge(model.start, d2, 0.5)
model.add_edge(d1, d1, 0.9)
model.add_edge(d1, d2, 0.1)
model.add_edge(d2, d1, 0.1)
model.add_edge(d2, d2, 0.9)

sequence = 'CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC'
X = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in sequence]])
print(X.shape)

y_hat = model.predict(X)

print("sequence: {}".format(''.join(sequence)))
print("hmm pred: {}".format(''.join([str(y.item()) for y in y_hat[0]])))