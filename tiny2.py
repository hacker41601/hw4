import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as functional
#Author: Monynich Kiem
#Date: 04/2022
#Purpose: Tiny Shakespeare - Use an RNN to output text generation with Shakespeare as the training devide
#Main reference: Dr. Harrison's code as given to us in lecture
#other references:
#https://www.kaggle.com/code/sumantindurkhya/text-generation-from-shakespeare-s-play-pytorch/notebook
#https://www.youtube.com/watch?v=xs6dOWlpQbM
#https://opendatascience.com/optimizing-pytorch-performance-batch-size-with-pytorch-profiler/

#hyperparamters
epochs = 30
batch = 64

#read and split vocab------------------------------------------------------------------------------
#read in
file = open("tiny-shakespeare.txt", "r").read()
#extract characters
characters = list(set(file))
#vocab section
intChar = dict(enumerate(characters))
charInt = {character: index for index, character in intChar.items()}
#print(intChar)
vocab_size = len(charInt)

#functions--------------------------------------------------------------------------------
def create_one_hot(sequence, vocab_size):
    #defines a matrix of vocab_size with all 0's so use np.zeros
    #dim = batch size x seq lenth x vocab size
    encoding = np.zeros((1, len(sequence), vocab_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
    return encoding

#rnn class
class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    #Define how inputs translate into outputs
    def forward(self, x):
        #hidden_state = self.init_hidden()
        output, hidden_state = self.rnn(x)
        #take off one to deal with extra dim from batch
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden_state
        
    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden

def predict(model, character):
    character_input = np.array([charInt[c] for c in character])
    character_input = create_one_hot(character_input, vocab_size)
    character_input = torch.from_numpy(character_input)
    out, hidden = model(character_input)

    prob = nn.functional.softmax(out[-1], dim=0).data
    character_index = torch.max(prob, dim=0)[1].item()

    return intChar[character_index], hidden
    
def sample(model, out_len, start='QUEEN:'):
    characters = [ch for ch in start]
    current_size = out_len - len(characters)
    for i in range(current_size):
        character, hidden_state = predict(model, characters)
        characters.append(character)

    return ''.join(characters)

#implementation----------------------------------------------------------------------------
model = RNNModel(vocab_size, vocab_size, 500, 2) #only 1 layer is too simplistic and causes lots of repeats for some reason or 100 in seq length

#define loss
loss = nn.CrossEntropyLoss()

#use Adam again
optimizer = torch.optim.Adam(model.parameters())

#initialize variables
input_sequence = []
target_sequence = []
sentences = []

#split into segments that can be used
segments = [file[pos:pos+51] for pos, i in enumerate(list(file)) if pos % 51 == 0]
#combine every 4 segments, of length 42, into length 100
print(len(segments)) #11154 for 100, 25667 for 42, 16166 for 69, anything >= 51 works 21871 the loss for 51 is really low at 2.5 ish where as 42 and 100 are at 3.4 smth
new_segment = ""
for i in range(len(segments)):
    new_segment += segments[i]
    if i % 4 == 3:
        sentences.append(new_segment)
        new_segment = ""
        
#shifting sequences by 1, similar to the pseudocode where First became irst so it can predict the next one
for i in range(len(sentences)):
    input_sequence.append(sentences[i][:-1])
    target_sequence.append(sentences[i][1:])
    
#constructing the one hots, replace all chars with ints
for i in range(len(sentences)):
    input_sequence[i] = [charInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charInt[character] for character in target_sequence[i]]
    #converting target_seq into a tensor, for loss only need the int output

#input sequences into one-hots
for i in range(len(input_sequence)):
    input_sequence[i] = create_one_hot(input_sequence[i], vocab_size)

# Batch data
#/Users/monynichkiem/Desktop/hw4/tiny2.py:104: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at  /Users/distiller/project/pytorch/torch/csrc/utils/tensor_new.cpp:210.)

#https://opendatascience.com/optimizing-pytorch-performance-batch-size-with-pytorch-profiler/
input_tensor = torch.FloatTensor(input_sequence)
input_tensor = torch.reshape(input_tensor, (len(input_tensor), len(sentences[0])-1, vocab_size))
training = TensorDataset(input_tensor, torch.FloatTensor(target_sequence))
trainLoader = DataLoader(training, batch_size=batch)

for epoch in range(epochs):
    print("Epoch:", epoch)
    for x, y in trainLoader:
        optimizer.zero_grad()
        output, hidden = model(x)
        lossValue = loss(output, y.view(-1).long())
        lossValue.backward()
        optimizer.step()
        
    print("Loss of This Epoch: {:.4f}".format(lossValue.item()))
    
print(sample(model, 100))
print("Final Loss: {:.4f}".format(lossValue.item()))

