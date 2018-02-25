# This is the main file that will contain all the encoder and decoder model
from __future__ import print_function

from parser import *
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
from keras import backend as K

latent_dim = 128
train_story , train_query , train_response , max_len_story , max_len_query , max_len_response , train_input_response = extract_text_data("dataset.txt")
batch_size = 32
epochs = 2
# create a vocabulary

# to create a better vocab , we need to extend the user utterance and bot utterance 

vocab_data = list()
vocab_data.extend(train_query)			# add all train queries to the vocab data
vocab_data.extend(train_response)		# add all train responses to the vocab data

word_to_idx = dict()
word_to_idx = create_vocabulary(vocab_data)	# create the vocabulary for the training data using the following functions


# These 3 vectorized versions of the code will help us build the model for task 1
vectorized_query = vectorize_utterance(train_query,max_len_query,word_to_idx)
vectorized_response = vectorize_utterance(train_response,max_len_response,word_to_idx)
vectorized_input_response = vectorize_utterance(train_input_response,max_len_response,word_to_idx)
vectorized_story = vectorize_utterance(train_story,max_len_story,word_to_idx)

vocab_size = len(word_to_idx)

inputs_train = vectorized_story 
queries_train = vectorized_query
answers_input_train = vectorized_input_response
answers_train = to_categorical(vectorized_response,num_classes=vocab_size)

# Some important information

print(" shape of inputs_train ")
print(inputs_train.shape)

print(" shape of queries_train ")
print(queries_train.shape)

print(" Size of Vocabulary is ")
print(vocab_size)

print(" Max lenght of story is ")
print(max_len_story)

print(" Max lenght of query is ")
print(max_len_query)

print(" Max lenght of response is ")
print(max_len_response)

print(" answers_input_train  is ")
print(answers_input_train)

# Create a idx_to_word dictionary
idx_to_word = dict()
for c,i in word_to_idx.items() :
	idx_to_word[i] = c

"""
for i,c in idx_to_word.items() :
	print(" Key %d ===> Value %s "%(i,c))
"""

# Now time to create the model

# placeholders
input_sequence = Input(shape=(max_len_story,))
question = Input(shape=(max_len_query,))
decoder_inputs_raw = Input(shape=(max_len_response,))
# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=latent_dim,
                              input_length=max_len_story))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=max_len_query,
                              input_length=max_len_story))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=latent_dim,
                               input_length=max_len_query))
question_encoder.add(Dropout(0.3))

decoder_sequence = Sequential()
decoder_sequence.add(Embedding(input_dim=vocab_size,
							output_dim=latent_dim,
							input_length=max_len_response))
decoder_sequence.add(Dropout(0.3))

# output: (samples, query_maxlen, embedding_dim)

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`


match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
#response = add([match,question_])
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.

encoder = LSTM(latent_dim,return_sequences = True,return_state = True)

encoder_outputs , state_h , state_c = encoder(answer)


encoder_states = [state_h , state_c]
print(" the type of encoder_states are ")
print(type(encoder_states))
print(type(state_h))
print(type(state_c))

decoder_inputs = decoder_sequence(decoder_inputs_raw)
print(" shape of decoder inputs ")
print(decoder_inputs.shape)
print(type(decoder_inputs))

decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)

decoder_outputs , _ , _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

decoder_dense = Dense(vocab_size,activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([input_sequence,question,decoder_inputs_raw],decoder_outputs)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#model.summary()
#iyt = input(" Program Paused ")

model.fit([inputs_train,queries_train,answers_input_train],answers_train,batch_size=batch_size,epochs=epochs,validation_split=0.2)




encoder_model = Model([input_sequence,question],encoder_states)


decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))


decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs , state_h , state_c = decoder_lstm(decoder_inputs,initial_state=decoder_states_inputs)

decoder_states = [state_h,state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs_raw] + decoder_states_inputs,[decoder_outputs] + decoder_states)



def decode_sequence(input_sequence) :
	
	states_value = encoder_model.predict(input_sequence)
	
	list_of_words = ['<begin>']
	target_sequence = vectorize_utterance([list_of_words],max_len_response,word_to_idx)
	stop_condition = False

	decoded_sentence = ''
	while not stop_condition :
		output_tokens , h , c = decoder_model.predict([target_sequence] + states_value)
		sampled_token_index = np.argmax(output_tokens[0,-1,:])
		sampled_char = idx_to_word[sampled_token_index]

		decoded_sentence += ' ' + sampled_char
		list_of_words += [sampled_char]

		if sampled_char == '<end>' or len(decoded_sentence) > max_len_response :
			stop_condition = True

		target_sequence = vectorize_utterance([list_of_words],max_len_response,word_to_idx)
		states_value = [h , c]

	return decoded_sentence



my_story = ['<begin>']
"""
while(1) :
	user_utterance_raw = input(" ask the bot something : ")

	my_vec_story = vectorize_utterance([my_story],max_len_story,word_to_idx)
	
	user_utterance = user_utterance_raw.lower().strip().split()		# split the user utterance into lower case words without any escape characters
	
	user_input = vectorize_utterance([user_utterance],max_len_query,word_to_idx)

	print(" My story is ")
	print(my_story)
	print(my_vec_story[0])
	print(my_vec_story[0].shape)
	print(type(my_vec_story[0]))
	#print(train_story[i].shape)
	print(" My query is ")
	print(user_utterance)
	print(user_input[0])
	print(user_input[0].shape)
	print(type(user_input[0]))

	bot_utterance = decoder_sequence([my_vec_story[0].reshape(1,86),user_input[0].reshape(1,19)])
	print(" bot utterance ")
	print(bot_utterance)

	
	story_to_append = story.copy()			# this is an important step, don't change it 

	#print(" The Train stoy till now is ")
	#print(train_story)

	my_story.append(story_to_append)		# append the final list of story to train_story

	story.extend(user_utterance)			# update the story with latest user utterance
	story.extend(bot_utterance)				# update the story with latest bot utterance
"""
for i in range(9) :
	print(" My story is ")
	print(train_story[i])
	print(inputs_train[i])
	print(inputs_train[i].shape)
	print(type(inputs_train[i]))
	#print(train_story[i].shape)
	print(" My query is ")
	print(train_query[i])
	print(queries_train[i])
	print(queries_train[i].shape)
	print(type(queries_train[i]))
	#print(train_query.shape)
	d_sentence = decode_sequence([inputs_train[i].reshape(1,86),queries_train[i].reshape(1,19)])
	print(" bot utterance is ")
	print(d_sentence)