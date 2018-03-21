# This is the main file that will contain all the encoder and decoder model
from __future__ import print_function

from parser_4ab import *
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM , Lambda
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
from keras import backend as K
import tensorflow as tf

latent_dim = 128
train_story , train_query , train_response , max_len_story , max_len_query , max_len_response , train_input_response = extract_text_data("dataset.txt")
batch_size = 32
epochs = 120
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
vectorized_story , max_lines_story , max_len_story = vectorize_story(train_story,max_len_query,max_len_response,word_to_idx)

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


def Reduce_Sum(x) :
	return K.tf.reduce_sum(x,axis=1)
# Now time to create the model

# placeholders
input_sequence = Input(shape=(max_lines_story,))

#sentence_inputs = [Input(shape=(max_len_story,)) for i in range(max_lines_story)]
input_embedder_m = Embedding(input_dim=vocab_size,output_dim=latent_dim,input_length=max_lines_story)
#input_embeddings_m = [ input_embedder_m(inp) for inp in input_sequence ]
input_encoded_m = input_embedder_m(input_sequence)
#input_encoded_m = tf.convert_to_tensor(input_embeddings_m)
#input_encoded_m = tf.reduce_sum(input_encoded_m_raw,axis=1)

#sentence_inputs = [Input(shape=(max_len_story,)) for i in range(max_lines_story)]
input_embedder_c = Embedding(input_dim=vocab_size,output_dim=max_len_query,input_length=max_lines_story)
#input_embeddings_c = [ input_embedder_m(inp) for inp in input_sequence ]
input_encoded_c= input_embedder_c(input_sequence)
#input_encoded_c = tf.convert_to_tensor(input_embeddings_c)
#input_encoded_c = tf.reduce_sum(input_encoded_c_raw,axis=1)

question = Input(shape=(max_len_query,))

decoder_inputs_raw = Input(shape=(max_len_response,))
# encoders
# embed the input sequence into a sequence of vectors
#input_encoder_m = Sequential()
#input_encoder_m.add(Embedding(input_dim=vocab_size,
#                             output_dim=latent_dim,
#                              input_length=max_len_story))
#input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
#input_encoder_c = Sequential()
#input_encoder_c.add(Embedding(input_dim=vocab_size,
#                              output_dim=max_len_query,
#                              input_length=max_len_story))
#input_encoder_c.add(Dropout(0.3))
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
#print(" Shape of sentence_inputs ")
#print(tf.convert_to_tensor(sentence_inputs).shape)
#print(tf.convert_to_tensor(sentence_inputs))

#input_encoded_m = input_encoder_m(input_sequence)

#input_encoded_c = input_encoder_c(input_sequence)
question_encoded_raw = question_encoder(question)
question_encoded = Lambda(Reduce_Sum(question_encoded_raw))
print(" question is encoded")
print(type(question_encoded))
# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`


print(" Printing all the shapes ")

print(" Shape of input_encoded_m")
print(input_encoded_m.shape)

print(" Shape of input_encoded_c ")
print(input_encoded_c.shape)

print(" Shape of question encoded ")
#print(question_encoded.shape)


match = dot([input_encoded_m, question_encoded],axes=(2,1))
print(" Shape of match before activation ")
print(match.shape)
match = Activation('softmax')(match)
print(" Shape of match after activation ")
print(match.shape)


def multiply_tensor(x,y) :
	return K.tf.multiply(K.transpose(x),y)
# add the match matrix with the second input vector sequence
#response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Lambda(multiply_tensor(match,input_encoded_c))
#response = add([match,question_])
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)


print(" Shape of response ")
print(response.shape)
print(tf.shape(response)[0])
print(tf.shape(response)[1])
print(tf.shape(response)[2])

#response_refined = tf.transpose(tf.reduce_sum(response,axis=0))

#print(" Shape of response_refined")
#print(response_refined.shape)
# concatenate the match matrix with the question vector sequence
answer = concatenate([K.reshape(response,[max_lines_story,latent_dim]), K.reshape(question_encoded,[1,latent_dim])],axis=0)
#answer = add([response_refined,question_encoded])

print(" Shape of answer ")
print(answer.shape)
# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.

encoder = LSTM(latent_dim,return_sequences = True,return_state = True,input_shape=(1,latent_dim))


print(" Shape of answer ")
print(answer.shape)

encoder_outputs , state_h , state_c = encoder(K.reshape(answer,[K.shape(response)[0],answer.shape[0],answer.shape[1]]))

encoder_states = [state_h , state_c]

decoder_inputs = decoder_sequence(decoder_inputs_raw)

decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)

decoder_outputs , _ , _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

decoder_dense = Dense(vocab_size,activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([input_sequence,question,decoder_inputs_raw],decoder_outputs)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

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