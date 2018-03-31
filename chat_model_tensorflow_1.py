# This is the main file that will contain all the encoder and decoder model
from __future__ import print_function

from parser_4ab import *
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM , Lambda , Reshape , Concatenate , multiply
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.engine import Layer
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
epochs = 12
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

# Create a idx_to_word dictionary
idx_to_word = dict()
for c,i in word_to_idx.items() :
	idx_to_word[i] = c

# All of the below functions are for custom functions that need to be implemented

		
def custom_concat(x) :
	y0 = tf.Print(x[0])
	y1 = tf.Print(x[1])
	return tf.concat([y0,y1],axis=1)

def Reduce_Sum_0(x) :
	return K.tf.reduce_sum(x,axis=0)

def Reduce_Sum_1(x) :
	return K.tf.reduce_sum(x,axis=1)


def Reduce_Sum_2(x) :
	return K.tf.reduce_sum(x,axis=2)

def Reduce_Sum_2_x(x) :
	return K.tf.reduce_sum(x,axis=2)

def custom_tensordot(x) :
	return tf.tensordot(x[0],x[1],axes=(2,2))


def my_transpose(x) :
	return tf.transpose(x,perm=[0,2,1])

def multiply_tensor(x) :
	#y1 = tf.tile(x[1],128)
	print(" Rank of x[1] is ")
	print(tf.rank(x[1]))
	one_vector = tf.ones([1,128],tf.float32)
	expanded_vector = tf.matmul(x[1],one_vector)
	f = tf.multiply(x[0],expanded_vector)
	print(" Shape of multiplication ")
	print(f.shape)
	return f


def output_of_lambda(input_shape) :
	print(" output shape of lambda is ")
	print(input_shape)
	return input_shape
    

def custom_transpose(x) :
	return tf.transpose(x,perm=[0,2,1])

def custom_permute(x,mask=None) :
	print(" Permuting here !!!! ")
	f = K.permute_dimensions(x,(0,2,1))
	print(" Shape of f ")
	print(f.shape)
	return f

def my_transpose_rev(x) :
	return tf.transpose(x,perm=[0,2,1])

def compute_mask(self,input,input_mask=None) :
	return [None,None]



# Now time to create the model

# placeholders

input_sequence = tf.keras.Input(shape=(max_lines_story,max_len_story,))

input_embedder_m = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=latent_dim,input_length=max_lines_story)

input_encoded_m_raw = input_embedder_m(input_sequence)

input_encoded_m = tf.reduce_sum(input_encoded_m_raw,axis=2)
#input_encoded_m = Lambda(Reduce_Sum_2)(input_encoded_m_raw)



input_embedder_c = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=latent_dim,input_length=max_lines_story)

input_encoded_c_raw = input_embedder_c(input_sequence) # Embedd a 8X19 sentences to 8X19X128 vectors

input_encoded_c = tf.reduce_sum(input_encoded_c_raw,axis=2)
#input_encoded_c = Lambda(Reduce_Sum_2)(input_encoded_c_raw) # Convert the 8X19X128 vector to 8X128 vectors


# embed the question into a vector of dimension of 128

question = tf.keras.Input(shape=(max_len_query,))

question_encoder = tf.keras.layers.Embedding(input_dim=vocab_size,
                               output_dim=latent_dim,
                               input_length=max_len_query)


question_encoded_raw = question_encoder(question) # convert a 1x19 vector to 1x19x128

question_encoded = tf.reduce_sum(question_encoded_raw,axis=1)
#question_encoded = Lambda(Reduce_Sum_1)(question_encoded_raw) # convert 1X19X128 to 1X128 vector

# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`



#question_encoded_reshaper = Reshape(target_shape=(1,question_encoded.get_shape().as_list()[1])) # reshape from ?X128 to 1X128
#question_encoded = question_encoded_reshaper(question_encoded)

question_encoded = tf.reshape(question_encoded,[tf.shape(question_encoded)[0],1,question_encoded.get_shape().as_list()[1]])

print(" Shape of question encoded is ")
print(question_encoded.shape)

print(" shape of input_encoded_m ")
print(input_encoded_m.shape)
#match = tf.tensordot(question_encoded,input_encoded_m,axes=(2,2))
match = tf.keras.layers.dot([question_encoded,input_encoded_m],axes=(2,2))
#match = Lambda(custom_tensordot)([question_encoded,input_encoded_m])	# compute the dot between question ans input_encoded to get probabilities of 8X1
print(" Shape of match before reshape is ")
print(match.shape)
#match = tf.reshape(match,[tf.shape(match)[0],1,match.get_shape().as_list()[3]])
match = tf.nn.softmax(match)

print(" shape of match ")
print(match.shape)
response = tf.keras.layers.dot([match,input_encoded_c],axes=(2,1)) 
print(" shape of response before reshape ")
print(response.shape)

#response = tf.reshape(response,[tf.shape(response)[0],1,response.get_shape().as_list()[3]])
print(" shape of response ")
print(response.shape)
# concatenate the match matrix with the question vector sequence
answer = tf.keras.layers.concatenate([response,question_encoded],axis=1)
print(" shape of answer ")
print(answer.shape)
# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.

encoder = tf.keras.layers.LSTM(latent_dim,return_sequences = True,return_state = True,input_shape=(1,latent_dim))

print(" Shape of response ")
print(K.shape(response))
_ , state_h_res , state_c_res = encoder(response)
encoder_outputs , state_h , state_c = encoder(question_encoded,initial_state=[state_h_res,state_c_res])

encoder_states = [state_h , state_c]

decoder_inputs_raw = tf.keras.Input(shape=(max_len_response,))

decoder_sequence = tf.keras.layers.Embedding(input_dim=vocab_size,
							output_dim=latent_dim,
							input_length=max_len_response)

decoder_inputs = decoder_sequence(decoder_inputs_raw)

decoder_lstm = tf.keras.layers.LSTM(latent_dim,return_sequences=True,return_state=True)

decoder_outputs , _ , _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(vocab_size,activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

print("   I have reached the end here    !!!! ")

model = tf.keras.models.Model([input_sequence,question,decoder_inputs_raw],decoder_outputs)

print(" Model Summary ")
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#model.summary()
#iyt = input(" Program Paused ")

model.fit([inputs_train,queries_train,answers_input_train],answers_train,batch_size=batch_size,epochs=epochs,validation_split=0.2)


print(type(encoder_states[0]))

encoder_model = tf.keras.models.Model([input_sequence,question],[encoder_states[0],encoder_states[1]])


decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))


decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs , state_h , state_c = decoder_lstm(decoder_inputs,initial_state=decoder_states_inputs)

decoder_states = [state_h,state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = tf.keras.models.Model([decoder_inputs_raw] + decoder_states_inputs,[decoder_outputs] + decoder_states)



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