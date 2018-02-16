# This is the first ever attempt at building the memory for memory network
# Here are some basic facts for the memory
# It is not a limited memory

import numpy as np
class Memory(object):
	"""docstring for Memory"""
	def __init__(self):
		self.memory = list()
		print("Memory created")

	def add_memory(self,new_memory):
		self.memory.append(new_memory)

	def update_memory(self,parameters) :
		# In this function , each memory segment is updated according some parameters given by
		# the user
		for mem in self.memory :
			generalize(mem,parameters)	# generalize is a function that is not yet defined but
										# has to be defined in this class only

	def receive_memory(self,input_vector):
		# The objective of this function is to score each memory with the parameters
		# The score function returns a score with the memory and both the
		# The memory with the highest score is returned
		for mem in self.memory :
			
			most_relevant_mem = None
			# Step:1 
			# Find the word embedding , here it is bag of words representation
			# embedded_word = bag_of_word_representation(mem)

			# Step:2
			# Find the cosine similarity
			# cosine_sim = find_cosine_sim(input_vector,embedded_word)

			# Step:3
			# if cosine_sim > max_sim :
			#	max_sim = cosine_sim
			#	most_relevant_mem = embedded_word

			return most_relevant_mem

		for mem in self.memory :
			mem_score = score(mem,parameters)
			score_list.append(mem_score)

		score_matrix = np.array(score_list)

		index = np.argmax(score_matrix)

		return memory[index]

	def print_memory(self):

		for mem in self.memory :
			words = mem.split(" ")
			if(words[0].equals('api_call')) :
				print(" The bot called an API and the parameters are as follows :")
				print(" %s,%s,%s,%s"%(words[1],words[2],words[3]))

			print(mem)