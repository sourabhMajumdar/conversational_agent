## Conversational Agent
Co-authored by : ***Sourabh Majumdar*** <msourabh970320@gmail.com>
Co-authored by : ***Tejas Bhatia*** <ted.bhatia@gmail.com>
### Objective

We are trying to develop a general purpose conversational agent.

### Motivation

As Humanity is advancing towards newwer technology, it becomes important to develop Systems (Agents) that can assist Humans
to work on better technologies. One of the most important aspects of such an agent is the ability to communicate its thought and ideas to humans. We aim to develop system (Dialog Systems) that can do the same.

### Our Approach

We start small by developing a conversational agent for the student library. The objective of this agent is to answer student
queries related to library matters.

**It would have been easy, except that there is no publically available dataset to train a such an agent.**

So what our plan is to train a model on simmillar grounds and figure out a way to integrate it to our above role. We have chosen the Dialog State Tracking Challenge 2017 as our starting point. What we plan to do is shadow the work of Facebook and build up a chat corpus in the background for the library.

here is a photo of the basic architecture we are building.
![alt tag](http://i.imgur.com/nv89JLc.png)

### Current Work

We are currently developing the memory part of the Facebook Memory Network.

### How do we create a memory

The answer is embedding. As one is reading this document, it should be evident that we need some kind of representation of words.

We use a embedding layer to find the vectorized representation. In keras this is done by the following lines
```
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding

# embed the input sequence into a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,
                                  output_dim=64))
    input_encoder_m.add(Dropout(0.3))
    # output: (samples, story_maxlen, embedding_dim)

    # embed the input into a sequence of vectors of size query_maxlen
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
    input_encoder_c.add(Dropout(0.3))
    # output: (samples, story_maxlen, query_maxlen)

    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                   output_dim=64,
                                   input_length=query_maxlen))
    question_encoder.add(Dropout(0.3))
    # output: (samples, query_maxlen, embedding_dim)

    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

```
### Now How do we create a Dataset from the data which we have

I suggest you look at the parser.py file to figure out how, its being done. To give a breif description, it extracts the words and vectorizes it according to the dictionary of words that we create from the training data.

First we separated the user utterances and the bot utterances, then we created a vocabulary of words used in both of these utterances. Then we proceeed to transform each utterance into a number representation where each word is converted to the corresponding index it appears in the aformentioned dictionary.

Creating the story is a little tricky but not that difficult. Each story is a set of user and bot utterances that has happened upto that point. We just add both of these sentences together to form a list of words that have been spoken untill now.
Now as we did with the user and bot utterances, we convert the story into a vectorized representation of the words.

### Start Small but do it right

Since we, have just created a dataset for only task 1 our next objective is to expand it to all the six tasks upto this point.
But before we do that, we must check if our model can learn to atleast perform task 1 reasonably well.
Hence now we must start to build a model for task 1.

the model is created in the file named ***chat_model.py*** and it has the code for memory network and the decoder model.
It is still half-baked and we yet have to solve the problem that is arriving with the decoder in inference model.

