from Bio import SeqIO  # pip install biopython
import argparse
import gzip
import numpy as np
import torch
import torch.nn as nn
from models import CNN
from torch.utils.data import DataLoader, TensorDataset
from models import DeepCpGModel
import matplotlib.pyplot as plt


def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.

    Parameters:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary with sequence IDs as keys and DNA sequences as values.
    """
    sequences = {}

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)
    else:
        with open(file_path, 'r') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)

    return sequences

def prepare_training_data(sequence_file: str, label_file: str):
    """
    Aligns nucleotide sequences with corresponding labels to create a training dataset.

    Parameters:
        sequence_file (str): Path to the FASTA file containing sequences.
        label_file (str): Path to the FASTA file containing labels.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains a DNA sequence and its label.
    """
    sequences = parse_fasta_file(sequence_file)
    labels = parse_fasta_file(label_file)

    if sequences.keys() != labels.keys():
        raise ValueError("Mismatch between sequence IDs and label IDs in the provided files.")

    return [(sequences[seq_id], labels[seq_id]) for seq_id in sequences]

def loss_fn(outputs, labels):
    """
    Computes the loss between predicted and true CpG island windows.

    Parameters:
        outputs (torch.Tensor): Predicted start and end indices of CpG islands. Shape: [batch_size, 2].
        labels (torch.Tensor): Ground truth start and end indices of CpG islands. Shape: [batch_size, 2].

    Returns:
        torch.Tensor: Normalized loss across the batch.
    """
    # Extract predicted and true start/end indices
    pred_start, pred_end = outputs[:, 0], outputs[:, 1]
    true_start, true_end = labels[:, 0], labels[:, 1]

    # Clamp predictions to valid bounds
    pred_start = torch.clamp(pred_start, 0, float('inf'))
    pred_end = torch.clamp(pred_end, 0, float('inf'))

    # Calculate loss components
    start_loss = torch.abs(pred_start - true_start)  # Difference in start indices
    end_loss = torch.abs(pred_end - true_end)        # Difference in end indices

    # Penalize invalid predictions (start > end)
    ordering_penalty = torch.relu(pred_start - pred_end)

    # Total loss
    total_loss = start_loss + end_loss + ordering_penalty

    # Normalize by batch size
    return total_loss.mean()





def train_classifier(training_data, dev_data, architecture='CNN', input_dim=8000, batch_size=16, num_epochs=100, learning_rate=0.005):
    """
    Trains a deep neural network to predict CpG island windows in DNA sequences and evaluates on dev data.

    Parameters:
        training_data (list[tuple[str, str, np.array]]): Training data consisting of sequences, labels, and one-hot encodings.
        dev_data (list[tuple[str, str, np.array]]): Dev data consisting of sequences, labels, and one-hot encodings.
        input_dim (int): Dimension of the input feature vectors (default: 8000).
        batch_size (int): Batch size for training (default: 16).
        num_epochs (int): Number of training epochs (default: 10).
        learning_rate (float): Learning rate for the optimizer (default: 0.001).

    Returns:
        DeepCpGModel: Trained model.
    """
    # Initialize the model and optimizer
    if architecture == 'CNN':
        model = CNN(input_length=input_dim, output_dim=2)
    else:
        model = DeepCpGModel(input_dim=input_dim, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def prepare_data(data):
        """Prepare sequences and labels for DataLoader."""
        sequences = [item[2] for item in data]  # Extract one-hot encoded sequences
        labels = []
        for label in [item[1] for item in data]:
            try:
                start = label.index('C')
                end = len(label) - 1 - label[::-1].index('C')
                labels.append([start, end])
            except ValueError:  # No 'C' in the label
                labels.append([0, 0])  # Default to no CpG island
        sequences = torch.tensor(sequences, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return TensorDataset(sequences, labels)

    # Prepare training and dev data loaders
    train_dataset = prepare_data(training_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = prepare_data(dev_data)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    dev_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_sequences, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)  # Model predictions
            loss = loss_fn(outputs, batch_labels)  # Custom loss function
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on the dev set
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for batch_sequences, batch_labels in dev_loader:
                outputs = model(batch_sequences)
                loss = loss_fn(outputs, batch_labels)
                dev_loss += loss.item()

        dev_loss /= len(dev_loader)
        dev_losses.append(dev_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")

    # Plot the training and dev losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), dev_losses, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Dev Losses')
    plt.legend()

    # save the model
    torch.save(model.state_dict(), f'model_with_{architecture}_architecture_{num_epochs}_epochs.pth')

    # save thr plot
    plt.savefig(f'loss_plot_{architecture}_{num_epochs}_epochs.png')

    return model



def annotate_sequence(model, sequence):
    """
    Annotates a DNA sequence with CpG island predictions.

    Parameters:
        model (object): Your trained classifier model.
        sequence (str): A DNA sequence to be annotated.

    Returns:
        str: A string of annotations, where 'C' marks a CpG island region and 'N' denotes non-CpG regions.
    """
    model.eval()
    one_hot_seq = dna_to_one_hot(sequence)  # Predefined in `one_hot`
    one_hot_tensor = torch.tensor(one_hot_seq, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    with torch.no_grad():
        prediction = model(one_hot_tensor)
    start, end = map(int, prediction.squeeze().tolist())
    annotation = ''.join(['C' if start <= i <= end else 'N' for i in range(len(sequence))])
    return annotation


def annotate_fasta_file(model, input_path, output_path):
    """
    Annotates all sequences in a FASTA file with CpG island predictions.

    Parameters:
        model (object): A trained classifier model.
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file where annotations will be saved.

    Writes:
        A gzipped FASTA file containing predicted annotations for each input sequence.
    """
    sequences = parse_fasta_file(input_path)

    with gzip.open(output_path, 'wt') as gzipped_file:
        for seq_id, sequence in sequences.items():
            annotation = annotate_sequence(model, sequence)
            gzipped_file.write(f">{seq_id}\n{annotation}\n")

def one_hot(data):
    """
    Converts a list of DNA sequences to one-hot encoding.

    Parameters:
        data (list[tuple[str, str]]): A list of tuples where each tuple contains a DNA sequence and its label.

    Returns:
        list[tuple[str, str, np.array]]: A list of tuples where each tuple contains a DNA sequence, its label, and its one-hot encoding.
    """
    def dna_to_one_hot(sequence):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return np.array([mapping[nuc] if nuc in mapping else [0, 0, 0, 0] for nuc in sequence]).flatten()

    one_hot_encoded_data = []
    for seq, label in data:
        one_hot_seq = dna_to_one_hot(seq)
        one_hot_encoded_data.append((seq, label, one_hot_seq))

    return one_hot_encoded_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str, help="Path to the output FASTA file for saving predictions.")

    args = parser.parse_args()

    training_sequences_path = r"data/CpG-islands.2K.seq.fa.gz"
    training_labels_path = r"data/CpG-islands.2K.lbl.fa.gz"

    # Prepare training data and train model
    training_data = prepare_training_data(training_sequences_path, training_labels_path)
    train_data_with_one_hot = one_hot(training_data)
    proportion_to_train = 0.8
    training_set, dev_set = train_data_with_one_hot[:int(len(train_data_with_one_hot)*proportion_to_train)], train_data_with_one_hot[int(len(train_data_with_one_hot)*proportion_to_train):]
    # CNN_classifier = train_classifier(training_set, dev_set, architecture='CNN')
    MLP_classifier = train_classifier(training_set, dev_set, architecture='MLP')
    

    # Annotate sequences and save predictions
    #annotate_fasta_file(classifier, args.fasta_path, args.output_path)


