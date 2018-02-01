#from nltk.corpus import wordnet
from PyDictionary import PyDictionary
from data_utils import load_dialog_task, load_candidates
 

class nltkHelper(object):
	def __init__(self, data_dir, task_id):
		self.data_dir = data_dir
		self.task_id = task_id

		candidates, self.candid2indx = load_candidates(
			self.data_dir, self.task_id)
		self.n_cand = len(candidates)
		print("Candidate Size", self.n_cand)
		self.indx2candid = dict(
		    (self.candid2indx[key], key) for key in self.candid2indx)
		# task data
		self.trainData, self.testData, self.valData = load_dialog_task(
		    self.data_dir, self.task_id, self.candid2indx, False)
		self.data = self.testData
		self.banned_words = ["i", "the"]
		self.pyD = PyDictionary()

	def find_synonyms(self, word):
		return self.pyD.synonym(word)

	# def recursive_find(query, syn, index):
	# 	# find a way to implement the recursive function
	# 	for i, w in enumerate(query):
	# 		continue if (i == index)
	# 		syns = find_synonyms(w)

	# 		for syn in syns:
	# 			print(query)
	# 			query[i] = syn
	# 			recursive_find(query, syn, i+1)

	def generate_queries(self, file):
		self.data.sort(key=lambda x:len(x[0]),reverse=True)
		join_string = " "
		for i, (story, query, answer) in enumerate(self.data):
			for j, w in enumerate(query):
				temp_query = list(query)
				syns = self.find_synonyms(w)
				if w in self.banned_words:
					continue
				if syns == None:
					continue
				# Generate syns for each word and then write to file the changed query and answer every syn
				# Should be a recursion to make use of every synonym commbinations of every word
				for syn in syns:
					temp_query[j] = syn
					#print(syn)
					#print("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")
					file.write("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")

	def generate_answers(self, file):
		self.data.sort(key=lambda x:len(x[0]),reverse=True)
		join_string = " "
		for i, (story, query, answer) in enumerate(self.data):
			for j, w in enumerate(answer):
				temp_query = list(query)
				syns = self.find_synonyms(w)
				if w in self.banned_words:
					continue
				if syns == None:
					continue
				# Need to have a framework for generating answers
				# for syn in syns:
				# 	temp_query[j] = syn
				# 	#print(syn)
				# 	#print("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")
				# 	file.write("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")

	def generate_multi_dialogs(self, file):
		#self.data.sort(key=lambda x:len(x[0]),reverse=True)
		join_string = " "
		prev_story = ""
		prev_query = ""
		prev_answer = ""

		# To get the last story with this logic
		self.data.append([[], "", ""])
		for i, (story, query, answer) in enumerate(self.data):
			# Go the last story and print that if new story starts
			if story == []:
				total_str = ""
				num = 1
				question = False
				for j, w in enumerate(prev_story):
					num = w[-1][1:]
					append_str = join_string.join(w[:-2])

					# Get punctuations to loose space around
					append_str = append_str.replace(" ' ", "'")
					# It's a question
					if (w[-2] == "$u"):
						total_str = total_str + num + " " + append_str + "\t"
						question = True
					# It's an answer
					elif (w[-2] == "$r" and question):
						total_str = total_str + append_str + "\n"
						question = False
					# It's an option or a simple fact not accompanied by a question
					elif (w[-2] == "$r"):
						total_str = total_str + num + " " + append_str + "\n"
						question = False

				if (prev_query != ""):
					# Print final query and answer
					int_num = int(num) + 1
					total_str = total_str + str(int_num) + " " + join_string.join(prev_query) + "\t" + self.indx2candid[int(prev_answer)]
					# Get tag words to be in capital case
					total_str = total_str.replace("<silence>", "<SILENCE>")
					# check for other punctuations
					total_str = total_str.replace(" ' ", "'")
					# total_str = total_str.replace(" . ", ". ")
					# total_str = total_str.replace(" , ", ",")
					# total_str = total_str.replace(" : ", ": ")
					new_total_str = ""
					end_index = int_num * 2
					int_num = int_num + 1

					# Create the same sequence of story with sentence numbers differently, all merged into one story
					# Testing knowledge retention over multiple conversations, and inference capacity
					for sents in total_str.split("\n"):
						sent_splits = sents.split(" ")
						new_total_str = new_total_str + str(int_num) + " " + join_string.join(sent_splits[1:]) + "\n"
						int_num = int_num + 1

					total_str = total_str + "\n" + new_total_str	
					file.write(total_str + "\n")

			prev_story = story
			prev_query = query
			prev_answer = answer

	def generate_multi_dialogs_from_file(self, file, file_input):
		stories = file_input.split("\n\n")
		full_story = ""
		join_string = " "
		for i, story in enumerate(stories):
			if story == "":
				continue
			sents = story.split("\n")
			num = sents[-1].split(" ")[0]
			int_num = int(num)
			int_num = int_num + 1
			num = str(int_num)
			total_str = ""

			for j, sent in enumerate(sents):
				sent_split = sent.split(" ")
				total_str = total_str + num + " " + join_string.join(sent_split[1:]) + "\n"
				int_num = int_num + 1
				num = str(int_num)

			full_story = full_story + story + "\n" + total_str + "\n"

		file.write(full_story)

if __name__ == '__main__':
	task_id = 5
	data_dir = "data/1-1-QA-without-context/"

	gen_data = nltkHelper(data_dir, task_id)
	f = open(data_dir + "dialog-babi-task" + str(task_id) + "-full-dialogs-tst-OOV-dynamic.txt", "w")
	# gen_data.generate_multi_dialogs(f)

	f_in = open(data_dir + "dialog-babi-task" + str(task_id) + "-full-dialogs-tst-OOV.txt", "r")
	gen_data.generate_multi_dialogs_from_file(f, f_in.read())

	f.close()
	f_in.close()