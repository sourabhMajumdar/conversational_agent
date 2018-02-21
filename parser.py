
# This program extracts the data and returns 
# the story , query and expected response from the file specified by user
def extract_text_data(file_name) :
	
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
			story = [['<begin>']]
			continue
		else :
		
			if conversations[0][0] == '1':		# if the first character in the conversation is one then start a new story
				story = [['<begin>']]
		

			user_utterance_raw = conversations[0][2:]	# user utterance starts from the 3rd character
			bot_utterance_raw = conversations[1]		# bot utterance is the second element of conversations


		

			user_utterance = user_utterance_raw.lower().strip().split()		# split the user utterance into lower case words without any escape characters
			bot_utterance = bot_utterance_raw.lower().strip().split()		# split the bot utterance into lower case words without any escape characters

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

			story.append(user_utterance)			# update the story with latest user utterance
			story.append(bot_utterance)				# update the story with latest bot utterance


			#tyyr = input(" Program Paused !!")
	return (train_story,train_query,train_response)

	
def create_vocabulary(train_story) :

	vocab = set()
	for context in train_story :
		for line in context :
			for word in line :
				vocab.add(word)

	word_to_idx = dict()
	idx_to_word = list()
	for i,c in enumerate(sorted(vocab)) :
		word_to_idx[c] = i


	return (word_to_idx,word_to_idx)

train_story , train_query , train_response = extract_text_data("dataset.txt")
# create a vocabulary

word_to_idx , idx_to_word = create_vocabulary(train_story)

print(type(word_to_idx))
for c,i in word_to_idx.items() :
	print(" key :%s ,                              index :%d"%(c,i))

for i in range(len(train_story)) :
	print("Context till now ")
	print(train_story[i])
	print(" Train train_query ")
	print(train_query[i])
	print(" Train train_response ")
	print(train_response[i])

	egdn = input("Program Paused !!!")
	

