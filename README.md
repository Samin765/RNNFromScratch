# Recurrent Neural Network (RNN) Character-Level Text Generation

This project implements a simple character-level Recurrent Neural Network (RNN) to generate text based on an input dataset. The RNN is trained on a sequence of characters from a text file, and the model learns to predict the next character in the sequence.

## Features
- Customizable RNN parameters (number of hidden nodes, sequence length, learning rate, etc.)
- One-hot encoding of input and output characters
- Training via backpropagation through time (BPTT)
- Generates text after each training epoch
- Tracks loss across epochs

## Requirements
- MATLAB or Octave

## How it Works
The RNN is initialized with random weights and biases. It then processes sequences of characters and attempts to predict the next character in the sequence. The loss is computed using the softmax output layer and backpropagation is used to update the weights.

### Key Components
1. **Data Reading**: 
   - Reads the book text and creates character-to-index and index-to-character mappings.
2. **Character Encoding**:
   - Converts characters to one-hot vectors for input to the RNN.
3. **RNN Initialization**:
   - Initializes the weights and biases for the hidden state, input, and output layers.
4. **Training**:
   - The RNN is trained for a number of epochs using sequences of characters to minimize the loss.
   - After each epoch, the model generates a new sequence of characters.

### Code Structure
- `readData()`: Reads the text data and extracts unique characters.
- `getContainers()`: Creates mappings between characters and their indices.
- `char_to_vector()`: Converts a character into a one-hot encoded vector.
- `init_RNN()`: Initializes the RNN parameters (weights, biases).
- `vanilla_RNN()`: Runs the RNN forward pass to predict characters and compute loss.
- `RNNwGradient()`: Trains the RNN and tracks loss across epochs.
- `computeCost()`: Calculates the loss (cross-entropy).
- `sample_Index()`: Samples the next character from the softmax output.

## Parameters
- `m = 100`: Number of nodes in the hidden state
- `sig = 0.01`: Standard deviation for weight initialization
- `K`: Number of unique characters
- `x_0`: Initial input character
- `h_0`: Initial hidden state (all zeros)
- `n = 5`: Sequence length
- `epochs = 200`: Number of training epochs
