## Conversational Agent

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

I suggest you look at the parser.py file to figure out how, its being done. Remeber it is still half-baked and we still need to convert the words into their indices according to the dictionary.

