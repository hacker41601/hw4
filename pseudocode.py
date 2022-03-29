import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as functional
#extract characters
#file = open("tiny-shakespeare.txt", "r").read()
file = ["Mr. and Mrs.Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere."]
characters = set("".join(file))
#characters = list(set(file))
print(characters)
#set up vocab
int_to_char = dict(enumerate(characters))
print(int_to_char)

#inverse
charInt = {character: index for index, character in int_to_char.items()}
print(int_to_char.items())
print(charInt)

#shifting the word from First to irst
input_seq = []
target_seq = []
for i in range(len(file)):
    input_seq.append(file[i][:-1])
    target_seq.append(file[i][1:])

#constructing the one hots, replace all chars with ints
for i in range(len(file)):
    input_seq[i] = [charInt[character] for character in input_seq[i]]
    target_seq[i] = [charInt[character] for character in target_seq[i]]
#converting target_seq into a tensor, for loss only need the int output

#print(input_seq)
#print(target_seq)

vocab_size = len(charInt)
print(vocab_size)

def create_one_hot(sequence, vocab_size):
    #defines a matrix of vocab_size with all 0's os use np.zeros
    #dim = batch size x seq lenth x vocab size
    encoding = np.zeros((1, len(sequence), vocab_size), dtype = np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
    return encoding

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden_state = self.init_hidden()
        print(x.size())
        output, hidden_state = self.rnn(x, hidden_state)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        #fc stands for fully connected layer
        return output, hidden_state
        
    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden
        
model = RNNModel(vocab_size, vocab_size, 100, 1)

#Define loss
loss = nn.CrossEntropyLoss()

#Use Adam again
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(150):
    for i in range(len(input_seq)):
        optimizer.zero_grad()
        x = torch.from_numpy(create_one_hot(input_seq[i], vocab_size))
        print(x)
        y = torch.Tensor(target_seq[i])
        print(y)
        output, hidden = model(x)
        
        print(output)
        print(hidden)
        print(output.size())
        print(y.view(-1).long().size())
        lossValue = loss(output, y.view(-1).long())
        lossValue.backward()
        optimizer.step()
        
        print("Loss: {:4f}".format(lossValue.item()))

def predict(model, character):
    characterInput = np.array([charInt[c] for c in character])
    characterInput = create_one_hot(characterInput, vocab_size)
    characterInput = torch.from_numpy(characterInput)
    out, hidden = model(characterInput)
    
    prob = nn.functional.softmax(out[-1], dim = 0).data
    character_index = torch.max(prob, dim = 0)[1].item()
    
    return int_to_char[character_index], hidden
    
def sample(model, out_len, start = "The"):
    characters = [ch for ch in start]
    current_size = out_len - len(characters)
    for i in range(current_size):
        character, hidden_state = predict(model, characters)
        characters.append(character)
        
    return "".join(characters)
    
print(sample(model, 100))
        
        

        



        


