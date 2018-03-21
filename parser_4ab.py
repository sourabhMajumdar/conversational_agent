from keras.preprocessing.sequence import pad_sequences
# This program extracts the data and returns 
# the story , query and expected response from the file specified by user

def extract_text_data(file_name) :

	max_len_query = 0
	max_len_response = 0
	max_len_story = 0
	# open the file specified by user
	f_handle = open(file_name)
	
	train_story = list()			# list that holds the story for training 
	train_query = list()			# list that holds the query at that point, asked by the user
	train_response = list()			# list that holds the response uttered by the bot at that instant
	train_input_response = list()

	story = list()					# a temporary list that expands for every context of the story
	new_story = list()				# a temporary place holder for a temp list
	
	for line in f_handle :

		conversations = line.split('\t')	# split the conversation between the user and the bot as they are separated by a tab
		
		if conversations[0] == '\n':		# if encountered a new_line character then continue 
			# train a new story
			story = list()
			continue
		else :
		
			if conversations[0][0] == '1':		# if the first character in the conversation is one then start a new story
				story = list()
		

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

			input_bot_utterance = ['<begin>']
			input_bot_utterance.extend(bot_utterance)
			bot_utterance.extend(['<end>'])


			train_story.append(story_to_append)		# append the final list of story to train_story
			train_query.append(user_utterance)		# append the final list of user_utterance to train_query
			train_response.append(bot_utterance)	# append the final list of bot_utterance to train_response
			train_input_response.append(input_bot_utterance)

			#story.extend(user_utterance)			# update the story with latest user utterance
			#story.extend(bot_utterance)				# update the story with latest bot utterance
			story.append(user_utterance)
			story.append(bot_utterance)
			max_len_story = max(max_len_story,len(story))

			#tyyr = input(" Program Paused !!")
	return (train_story,train_query,train_response,max_len_story,max_len_query,max_len_response,train_input_response)

	
def create_vocabulary(vocab_data) :

	vocab = set()
	vocab.add('<begin>')
	for line in vocab_data :
		for word in line :
			vocab.add(word)

	word_to_idx = dict()
	for i,c in enumerate(sorted(vocab)) :
		word_to_idx[c] = i


	return word_to_idx

# This function generates a vectorized representation of all words and performs padding also
def vectorize_utterance(train_data,max_len_data,word_to_idx) :
	
	vectorized_data = list()
	
	for line in train_data :
		input_indices = [word_to_idx[w] for w in line]
		vectorized_data.append(input_indices)

	return pad_sequences(vectorized_data,maxlen=max_len_data)

def vectorize_story(train_story,max_len_query,max_len_response,word_to_idx) :

	
	
	vectorized_story = list()
	
	for sentences in train_story :
		
		vectorized_lines = list()
		max_lines_story = 0
		max_len_story = max(max_len_query,max_len_response)

		for line in sentences :
			
			my_custom_line = vectorize_utterance([line],max_len_story,word_to_idx)

			#input_indices = [word_to_idx[word] for word in line]

			vectorized_lines.append(my_custom_line[0].tolist())
			max_lines_story = max(max_lines_story,len(vectorized_lines))
			#vectorized_lines.append(input_indices)

		vectorized_story.append(vectorized_lines)

	return pad_sequences(vectorized_story,maxlen=max_lines_story) , max_lines_story , max_len_story
'''
train_story , train_query , train_response , max_len_story , max_len_query , max_len_response , train_input_response = extract_text_data("dataset.txt")

vocab_data = list()
vocab_data.extend(train_query)			# add all train queries to the vocab data
vocab_data.extend(train_response)		# add all train responses to the vocab data

word_to_idx = dict()
word_to_idx = create_vocabulary(vocab_data)	# create the vocabulary for the training data using the following functions

vectorized_story = vectorize_story(train_story,max_len_story,word_to_idx)

vectorized_query = vectorize_utterance(train_query,max_len_query,word_to_idx)
vectorized_response = vectorize_utterance(train_response,max_len_response,word_to_idx)
'''
'''print(len(train_story))
print(len(vectorized_story))
print(len(vectorized_story[0]))
bd = input(" Paused !!! ")
for i in range(len(train_story)) :
	
	print(" Printing story ")
	print(train_story[i])
	print(vectorized_story[i])
	
	print(" Printing query ")
	print(train_query[i])
	print(vectorized_query[i])
	
	uty = input(" Paused !!! ")
'''
#print(train_story[1])
#for story in train_story :
#	print(story)
#	pg = input(" Paused !! ")