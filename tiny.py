#Author: Monynich Kiem
#Purpose: Text generation using RNN (recurrent neural networks) to generate text using Tiny Shakespeare Text
#Date: 03/29/2022
#Libraries are allowed for this, the implementation of RNN is what we need to do on our own
#Following Youtube Tutorial?

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

file = open("tiny-shakespeare.txt", "r").read()
print("length of file: ", len(file)) #1115394
print(file[:250])

#get the unique characters and store into a dictionary
vocab = sorted(set(file))
vocab_size = len(vocab)
print("num of unique characters: ", vocab_size) #there are 65 unique characters
idx_char = np.array(vocab)
#print(idx_char) #prints an array of the unique characeters in the dictionary

#character to integer representation
char_idx = {unique: idx for idx, unique, in enumerate(vocab)}
text_int = np.array([char_idx[char] for char in file])
for char,_ in zip(char_idx, range(65)):
    print(" {:4s}: {:3d},".format(repr(char), char_idx[char]))
    #this prints out the mapping of which character is mapped to what integer
    #we have a dictionary of 65 unique character from 0-64
print("{} --> chars mapped to int --> {} ".format(repr(file[:13]), text_int[:13]))
#string to int vector representation sample

#feed the model some string of text and predicts what it will predict
sequence_length = 100
ex_per_epoch = len(file) // (sequence_length + 1)
#gets the slices of the text file Tiny Shakespeare using the text_int function that gets the characters in the file
char_dataset = tf.data.Dataset.from_tensor_slices(text_int)

for i in char_dataset.take(5): #prints out the word First
    print(idx_char[i.numpy()])

sequences = char_dataset.batch(sequence_length + 1, drop_remainder = True)
for item in sequences.take(5): #take the first 5 sequences
    print(repr("".join(idx_char[item.numpy()])))
    
def split_input_target(chunk):
    input_txt = chunk[:-1]
    target_txt = chunk[1:]
    return input_txt, target_txt

dataset = sequences.map(split_input_target)
for input_ex, target_ex in dataset.take(1):
    print("input data", repr("".join(idx_char[input_ex.numpy()])))
    #shift the data 1 character for the target data
    print("target data", repr("".join(idx_char[target_ex.numpy()])))
    
for i, (input_idx, target_idx) in enumerate(zip(input_ex[:5], target_ex[:5])):
    print("step {:4d}".format(i))
    print(" input{} ({:s})".format(input_idx, repr(idx_char[input_idx])))
    print(" prediction {} ({:s})".format(target_idx, repr(idx_char[target_idx])))
    #not trained yet, but it predicts what the next character might be
    
batch_size = 64
buffer_size = 10000

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder = True)
embedding_dim = 256
rnn_units = 1024

def rnn(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size, None]),
    tf.keras.layers.GRU(rnn_units, return_sequences = True, stateful = True, recurrent_initializer = "glorot_uniform"),
    tf.keras.layers.Dense(vocab_size)])
    return model
    
model = rnn(vocab_size = len(vocab), embedding_dim = embedding_dim, rnn_units = rnn_units, batch_size = batch_size)

model.summary()

for input_ex_batch, target_ex_batch in dataset.take(1):
    ex_batch_pred = model(input_ex_batch)
    print(ex_batch_pred.shape, "# batch side, sequence length, vocab size")

model.summary()

#Time to start training the model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

model.compile(optimizer = "adam", loss = loss)

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "chkpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_prefix, save_weights_only = True)

EPOCHS = 2
#history = model.fit(dataset, epochs = EPOCHS, callbacks = [checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = rnn(vocab_size, embedding_dim, rnn_units, batch_size = 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
#model.summmary()

def gen_txt(model, start_string):
    num_generate = 1000
    input_eval = [char_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    text_generated = []
    temperature = 1.0
    
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictiions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1, 0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_char[predicted_id])
        
    return (start_string + "".join(text_generated))

print(gen_txt(model, start_string = "ROMEO: "))
