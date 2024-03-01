import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ...utils.printing import message

class Encoder():
    """ Abstract class for one hot encoding of sequences.

    Parameters
    ----------
    alphabet (str) : Alphabet of the sequences to encode.    
    """

    def __init__(self, alphabet, verbose=False) -> None:
        """ Initialize the encoder with the alphabet of the sequences to encode. 
        
        Variables
        ---------
        encoder (OneHotEncoder)   : OneHotEncoder object from sklearn. It encodes categorical features as a one-hot numeric array.
        alphabet (str)            : Alphabet of the sequences to encode.
        """
        self.encoder = OneHotEncoder(categories=[list(alphabet)], sparse_output=False, dtype=int, handle_unknown='ignore')
        self.encoder.fit(np.array(list(alphabet)).reshape(-1, 1))
        self.alphabet = alphabet
        self.verbose = verbose

    def encode_sequence(self, sequence, max_sequence_length=None):
        """ Helper function to one hot encode one sequence.
        When sequence length is smaller than max_sequence_length, the sequence is padded with zeros at the tail.
        
        Parameters
        ----------
        sequence (str) : Sequence to encode.
        max_sequence_length (int) : length of the encoded sequence. If it is bigger than the length of the target sequence, the sequence is padded with zeros at the tail. Default: None, do not padd zeros.

        Returns
        -------
        one_hot_encoded (np.array) : One hot encoded sequence. It has shape (len(alphabet), max_sequence_length).
        """

        # set encoding length as the original sequence length, if not declared
        if max_sequence_length is None:
            max_sequence_length = len(sequence)

        # pad zeros to the tail if necessary
        sequence = sequence + ''.join(["?"]*(max_sequence_length - len(sequence)))

        # encode sequence
        reshaped_sequences = [[char.lower()] for char in sequence]
        one_hot_encoded = self.encoder.transform(reshaped_sequences).T

        return one_hot_encoded

    def one_hot_encode_batch(self, sequences_batch, max_sequence_length=None):
        """ Helper function to one hot encode a batch of sequences. 
        When sequence length is smaller than max_sequence_length, the sequence is padded with zeros at the tail.
        
        Parameters
        ----------
        sequences_batch (list) : List of sequences to encode.
        max_sequence_length (int)   : Maximun length of the sequences. If None, the maximum length of the current set of sequences is used. Default: None

        Returns
        -------
        encoded_batch (np.array) : One hot encoded batch of sequences. It has shape (len(sequences_batch), len(alphabet), max_sequence_length).
        """

        # set encoding length as the maximun of the batch, if not declared
        if max_sequence_length is None:
            max_sequence_length = max([len(sequence) for sequence in sequences_batch])

        # pad zeros to the tail if necessary
        sequences_batch = [sequence + ''.join(["?"]*(max_sequence_length - len(sequence))) for sequence in sequences_batch]

        # encode batch
        reshaped_sequences = [[char.lower()] for sequence in sequences_batch for char in sequence]
        encoded_batch = self.encoder.transform(reshaped_sequences)

        return encoded_batch

    def encode_sequences(self, sequences, batch_size = 1, max_sequence_length=None):
        """ Encode the sequences. Save encoded sequences if required.

        By default, batch_size = 1, so the sequences are encoded one by one.
        Optionally, when the batch_size is set to a value greater than 1, the encoding is done in batches to speed up the process.
        In this case, the sequences are encoded during each batch run. The encoded sequences are then concatenated and transposed to match the shape of the non-batched encoding.
        
        Parameters
        ----------
        sequences (list) : List of sequences to encode.
        batch_size (int) : Number of sequences to encode at once. Default: 1
        max_sequence_length (int)   : Maximun length of the sequences. If None, the maximum length of the current set of sequences is used. Default: None

        Returns
        -------
        encoded_sequences (np.array) : One hot encoded sequences. It has shape (len(sequences), len(alphabet), max_sequence_length).
        """
        # set encoding length as the maximun of the input sequences, if not declared
        if max_sequence_length is None:
            max_sequence_length = max([len(sequence) for sequence in sequences])

        # encode one by one
        if batch_size == 1:
            message("Starting to one hot encode", verbose=self.verbose)
            encoded_sequences = np.stack([self.encode_sequence(sequence) for sequence in sequences])

        # encode by batches
        else:
            message("Starting to one hot encode", verbose=self.verbose)
            encoded_batches = []

            # for each batch, encode the sequences
            for i in range(0, len(sequences), batch_size):
                message("++Processing batch ", str(i), verbose=self.verbose)
                batch = sequences[i:i + batch_size]
                encoded_batch = self.one_hot_encode_batch(batch, max_sequence_length)
                encoded_batches.append(encoded_batch)

            # stack one hot encoded batches
            encoded_sequences = np.concatenate(encoded_batches)
            encoded_sequences = np.transpose( encoded_sequences.reshape(-1, max_sequence_length, len(self.alphabet)) ,(0,2,1) )

        message("Finished one hot encoding", verbose=self.verbose)

        return encoded_sequences

    def decode_one_hot(self,encoded_sequence):
        decoded_sequence = ""
        encoded_sequence = encoded_sequence.reshape(4, -1)
        for one_hot_vector in encoded_sequence.T:
            # check if one hot vector is numpy array
            if not isinstance(one_hot_vector, np.ndarray):
                one_hot_vector = one_hot_vector.numpy()
            if np.sum(one_hot_vector) == 0:
                decoded_sequence += "N"
            else:
                index = np.argmax(one_hot_vector)
                if index < len(self.alphabet):
                    decoded_sequence += self.alphabet[index]
                else:
                    decoded_sequence += "N"
        # make to uppercase
        decoded_sequence = decoded_sequence.upper()
        return decoded_sequence

class DNAEncoder(Encoder):
    """ Custom class to use Encoder with DNA sequences. """

    def __init__(self, alphabet="acgt", verbose=False) -> None:
        super().__init__(alphabet, verbose=verbose)

    