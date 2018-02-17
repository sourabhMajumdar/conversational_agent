## Conversational Agent

### Objective

We are trying to develop a general purpose conversational agent.

### Motivation

As Humanity is advancing towards newwer technology, it becomes important to develop Systems (Agents) that can assist Humans
to work on better technologies. One of the most important aspects of such an agent is the ability to communicate its thought and ideas to humans. We aim to develop system (Dialog Systems) that can do the same.

### Our Approach

We start small by developing a conversational agent for the student library. The objective of this agent is to answer student
queries related to library matters.

*It would have been easy, except that there is no publically available dataset to train a such an agent.*

So what our plan is to train a model on simmillar grounds and figure out a way to integrate it to our above role. We have chosen the Dialog State Tracking Challenge 2017 as our starting point. What we plan to do is shadow the work of Facebook and build up a chat corpus in the background for the library.

### Current Work

We are currently developing the memory part of the Facebook Memory Network.

### How do we create a memory

The answer is embedding. As one is reading this document, it should be evident that we need some kind of representation of words.
We will use GloVe vectors for this and especialy the following two code snippents

*to load the glove vectors*
```
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
```

*to creating the embedding layer*

```
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
```
