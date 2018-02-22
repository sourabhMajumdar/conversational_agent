from keras.preprocessing.sequence import pad_sequences
# This program extracts the data and returns 
# the story , query and expected response from the file specified by user
max_len_query = 0
max_len_response = 0
max_len_story = 0


def extract_text_data(file_name) :

	global max_len_query,max_len_response,max_len_story
	# open the file specified by user
	f_handle = open(file_name)
	
	train_story = list()			# list that holds the story for training 
	train_query = list()			# list that holds the query at that point, asked by the user
	train_response = list()			# list that holds the response uttered by the bot at that instant

	story = list()					# a temporary list that expands for every context of the story
	new_story = list()				# a temporary place holder for a temp list
	
	for line in f_handle :

		conversations = line.split('\t')	# split the conversation between the user and the bot as they are separated by a tab
		
		if conversations[0] == '\n':		# if encountered a new_line character then continue 
			# train a new story
			story = ['<begin>']
			continue
		else :
		
			if conversations[0][0] == '1':		# if the first character in the conversation is one then start a new story
				story = ['<begin>']
		

			user_utterance_raw = conversations[0][2:]	# user utterance starts from the 3rd character
			bot_utterance_raw = conversations[1]		# bot utterance is the second element of conversations


		

			user_utterance = user_utterance_raw.lower().strip().split()		# split the user utterance into lower case words without any escape characters
					
			max_len_query = max(max_len_query,len(user_utterance))
			
			bot_utterance = bot_utterance_raw.lower().strip().split()		# split the bot utterance into lower case words without any escape characters
			
			max_len_response = max(max_len_response,len(bot_utterance))

			# set of test statements to check if the story , user utterance and bot utterance are according to expectations
			#print(" The story is ")
			#print(story)
			#print(" User :")
			#print(user_utterance)
			#print(" Bot :")
			#print(bot_utterance)

			story_to_append = story.copy()			# this is an important step, don't change it 

			#print(" The Train stoy till now is ")
			#print(train_story)


			train_story.append(story_to_append)		# append the final list of story to train_story
			train_query.append(user_utterance)		# append the final list of user_utterance to train_query
			train_response.append(bot_utterance)	# append the final list of bot_utterance to train_response

			story.extend(user_utterance)			# update the story with latest user utterance
			story.extend(bot_utterance)				# update the story with latest bot utterance
			max_len_story = max(max_len_story,len(story))

			#tyyr = input(" Program Paused !!")
	return (train_story,train_query,train_response)

	
def create_vocabulary(vocab_data) :

	vocab = set()
	vocab.add('<begin>')
	for line in vocab_data :
		for word in line :
			vocab.add(word)

	word_to_idx = dict()
	idx_to_word = list()
	for i,c in enumerate(sorted(vocab)) :
		word_to_idx[c] = i


	return (word_to_idx,word_to_idx)

# This function generates a vectorized representation of all words and performs padding also
def vectorize_utterance(train_data,max_len_data) :
	
	vectorized_data = list()
	
	for line in train_data :
		input_indices = [word_to_idx[w] for w in line]
		vectorized_data.append(input_indices)

	return pad_sequences(vectorized_data,maxlen=max_len_data)


train_story , train_query , train_response = extract_text_data("dataset.txt")
# create a vocabulary

# to create a better vocab , we need to extend the user utterance and bot utterance 

vocab_data = list()
vocab_data.extend(train_query)			# add all train queries to the vocab data
vocab_data.extend(train_response)		# add all train responses to the vocab data


word_to_idx , idx_to_word = create_vocabulary(vocab_data)	# create the vocabulary for the training data using the following functions



#print(type(word_to_idx))			# un-comment the following lines , if you want to see what the word to index dictionary looks like
#for c,i in word_to_idx.items() :
#	print(" key :%s ,                              index :%d"%(c,i))



# These 3 vectorized versions of the code will help us build the model for task 1
vectorized_query = vectorize_utterance(train_query,max_len_query)
vectorized_response = vectorize_utterance(train_response,max_len_response)
vectorized_story = vectorize_utterance(train_story,max_len_story)




# If one needs to see what every training data hold , one can un-comment the following code below
#print(" vectorize_utterance for train query is ")
#print(vectorized_query)
#print(" max length of query ")
#print(max_len_query)
#print(" vectorize_utterance for train response is ")
#print(vectorized_response)
#print(" vectorize_utterance for train story is ")
#print(vectorized_story)
#print(" max length of response ")
#print(max_len_response)
#print(" max len story is ")
#print(max_len_story)

#for i in range(len(train_story)) :
#	print("Context till now ")
#	print(train_story[i])
#	print(" Train train_query ")
#	print(train_query[i])
#	print(" Train train_response ")
#	print(train_response[i])

#	egdn = input("Program Paused !!!")
	

