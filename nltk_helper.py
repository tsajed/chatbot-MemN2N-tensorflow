from nltk.corpus import wordnet

def find_synonyms(word):
	return wordnet.synsets(word)


def generate_data(query, answer):
	# generate multiple queries depending on synsets
	queries = []
	for w in query:
		syns = find_synonyms
		# some kind of replacing algorithm
		queries.append(new_query + "\t" + answer)

	file.write(queries)