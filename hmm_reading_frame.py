import numpy as np
import annotate_cpg
import hmm_model

# Load training data
training_sequences_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.seq.fa.gz"
training_labels_path = r"C:\Users\roise\OneDrive\Desktop\Hebrew U\CBIO\ex2\data\CpG-islands.2K.lbl.fa.gz"
training_data = annotate_cpg.prepare_training_data(training_sequences_path, training_labels_path)
training_data = [pair for pair in training_data if all(char in "AGCT" for char in pair[0])]
train, test = training_data[:800], training_data[800:]

def calculate_gc_content(sequence):
    """Calculate the GC content percentage of a given nucleotide sequence."""
    if not sequence:
        return 0
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


def classify_gc_content(sequence):
    """Classify sequence as low GC (<50%) or high GC (>=50%)."""
    return calculate_gc_content(sequence) >= 0.5


class markov_reading_frame:
    def __init__(self, frame=20):
        self.frame = frame
        self.half_frame = frame // 2
        self.states = {}
        self.transitions = {}

    def add_state(self, state):
        self.states[state] = state


    def train(self, data):
        transition_dict = {"CC_HH": 0, "CC_HL": 0, "CC_LH": 0, "CC_LL": 0,
                           "CN_HH": 0, "CN_HL": 0, "CN_LH": 0, "CN_LL": 0,
                           "NC_HH": 0, "NC_HL": 0, "NC_LH": 0, "NC_LL": 0,
                           "NN_HH": 0, "NN_HL": 0, "NN_LH": 0, "NN_LL": 0}
        for seq, annotation in data:
            for i in range(1, len(annotation) - 1):
                before_range = seq[max(0, i - self.half_frame):i]
                after_range = seq[i + 1:min(len(seq), i + self.half_frame+1)]
                gc_before = "H" if classify_gc_content(before_range) else "L"
                gc_after = "H" if classify_gc_content(after_range) else "L"
                transition_dict[annotation[i-1] + annotation[i] + "_" + gc_before + gc_after] += 1

        total_from_C_HH = transition_dict["CC_HH"] + transition_dict["CN_HH"]
        total_from_C_HL = transition_dict["CC_HL"] + transition_dict["CN_HL"]
        total_from_C_LH = transition_dict["CC_LH"] + transition_dict["CN_LH"]
        total_from_C_LL = transition_dict["CC_LL"] + transition_dict["CN_LL"]
        total_from_N_HH = transition_dict["NC_HH"] + transition_dict["NN_HH"]
        total_from_N_HL = transition_dict["NC_HL"] + transition_dict["NN_HL"]
        total_from_N_LH = transition_dict["NC_LH"] + transition_dict["NN_LH"]
        total_from_N_LL = transition_dict["NC_LL"] + transition_dict["NN_LL"]

        self.transitions = {"CC_HH": transition_dict["CC_HH"] / total_from_C_HH, "CN_HH": transition_dict["CN_HH"] / total_from_C_HH,
                            "CC_HL": transition_dict["CC_HL"] / total_from_C_HL, "CN_HL": transition_dict["CN_HL"] / total_from_C_HL,
                            "CC_LH": transition_dict["CC_LH"] / total_from_C_LH, "CN_LH": transition_dict["CN_LH"] / total_from_C_LH,
                            "CC_LL": transition_dict["CC_LL"] / total_from_C_LL, "CN_LL": transition_dict["CN_LL"] / total_from_C_LL,
                            "NC_HH": transition_dict["NC_HH"] / total_from_N_HH, "NN_HH": transition_dict["NN_HH"] / total_from_N_HH,
                            "NC_HL": transition_dict["NC_HL"] / total_from_N_HL, "NN_HL": transition_dict["NN_HL"] / total_from_N_HL,
                            "NC_LH": transition_dict["NC_LH"] / total_from_N_LH, "NN_LH": transition_dict["NN_LH"] / total_from_N_LH,
                            "NC_LL": transition_dict["NC_LL"] / total_from_N_LL, "NN_LL": transition_dict["NN_LL"] / total_from_N_LL}

    def predict(self, seq):
        cur_state = 'N'
        pred_states = [cur_state]
        for i in range(1, len(seq)):
            before_range = seq[max(0, i - self.half_frame):i]
            after_range = seq[i + 1:min(len(seq[0]), i + self.half_frame+1)]
            gc_before = "H" if classify_gc_content(before_range) else "L"
            gc_after = "H" if classify_gc_content(after_range) else "L"
            cur_state = "C" if np.random.rand() < self.transitions[cur_state + "C_" + gc_before + gc_after] else "N"
            pred_states.append(cur_state)
        return ''.join(pred_states)


def loss_over_dataset(data, model):
    loss, tp, fn, fp, tn = 0, 0, 0, 0, 0
    for seq, annotation in data:
        pred_states = model.predict(seq)
        _loss, _tp, _fn, _fp, _tn = hmm_model.loss_over_sequence(annotation, pred_states)
        loss += _loss
        tp += _tp
        fn += _fn
        fp += _fp
        tn += _tn
    loss = loss / len(data)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return loss, recall, precision


# Train the model
model = markov_reading_frame()
model.train(train)

print(training_data[0][1])
print(model.predict(training_data[0][0]))

# Evaluate the model
print("Training loss, recall, precision:", loss_over_dataset(test, model))
