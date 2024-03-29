# Foundations of Sequence Models

In the fifth course of the Deep Learning Specialization, you will become familiar with sequence models and their exciting applications such as speech recognition, music synthesis, chatbots, machine translation, natural language processing (NLP), and more. 

## About

Discover recurrent neural networks, a type of model that performs extremely well on temporal data, and several of its variants, including LSTMs, GRUs and Bidirectional RNNs.

## Learning Objectives

- Define notation for building sequence models
- Describe the architecture of a basic RNN
- Identify the main components of an LSTM
- Implement backpropagation through time for a basic RNN and an LSTM
- Give examples of several types of RNN
- Build a character-level text generation model using an RNN
- Store text data for processing using an RNN
- Sample novel sequences in an RNN
- Explain the vanishing/exploding gradient problem in RNNs
- Apply gradient clipping as a solution for exploding gradients
- Describe the architecture of a GRU
- Use a bidirectional RNN to take information from two points of a sequence
- Stack multiple RNNs on top of each other to create a deep RNN
- Use the flexible Functional API to create complex models
- Generate your own jazz music with deep learning
- Apply an LSTM to a music generation task

## Notes

### RNNs

Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory." They can read inputs 𝑥⟨𝑡⟩ (such as words) one at a time, and remember some contextual information through the hidden layer activations that get passed from one time step to the next. This allows a unidirectional (one-way) RNN to take information from the past to process later inputs. A bidirectional (two-way) RNN can take context from both the past and the future

What you should remember:

- The recurrent neural network, or RNN, is essentially the repeated use of a single cell.
- A basic RNN reads inputs one at a time, and remembers information through the hidden layer activations (hidden states) that are passed from one time step to the next.
    - The time step dimension determines how many times to re-use the RNN cell
- Each cell takes two inputs at each time step:
    - The hidden state from the previous cell
    - The current time step's input data
- Each cell has two outputs at each time step:
    - A hidden state
    - A prediction

- Very large, or "exploding" gradients updates can be so large that they "overshoot" the optimal values during back prop -- making training difficult
    - Clip gradients before updating the parameters to avoid exploding gradients
- Sampling is a technique you can use to pick the index of the next character according to a probability distribution.
    - To begin character-level sampling:
        Input a "dummy" vector of zeros as a default input
        Run one step of forward propagation to get 𝑎⟨1⟩ (your first character) and 𝑦̂ ⟨1⟩ (probability distribution for the following character)
        When sampling, avoid generating the same result each time given the starting letter (and make your names more interesting!) by using np.random.choice


### Long Short-Term Memory (LSTM) Network

What you should remember:

- An LSTM is similar to an RNN in that they both use hidden states to pass along information, but an LSTM also uses a cell state, which is like a long-term memory, to help deal with the issue of vanishing gradients
- An LSTM cell consists of a cell state, or long-term memory, a hidden state, or short-term memory, along with 3 gates that constantly update the relevancy of its inputs:
    - A forget gate, which decides which input units should be remembered and passed along. It's a tensor with values between 0 and 1.
        - If a unit has a value close to 0, the LSTM will "forget" the stored state in the previous cell state.
        - If it has a value close to 1, the LSTM will mostly remember the corresponding value.
    - An update gate, again a tensor containing values between 0 and 1. It decides on what information to throw away, and what new information to add.
        - When a unit in the update gate is close to 1, the value of its candidate is passed on to the hidden state.
        - When a unit in the update gate is close to 0, it's prevented from being passed onto the hidden state.
    - And an output gate, which decides what gets sent as the output of the time step 



