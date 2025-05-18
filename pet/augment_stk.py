import pandas as pd
from tqdm import tqdm
import json
import random
from nltk.corpus import wordnet
import random
import numpy as np
import os


CLASS2AB = {'Task':'Task','Material':'Material','Technique': 'Technique', 'Process':'Process', 'Measure':'Measurement','Concept':'Term'}

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

def get_valid_words(line):
	line = line.replace("’", "")
	line = line.replace("'", "")
	line = line.replace("\t", " ")
	line = line.replace("\n", " ")	
	raw_words = line.split(' ')
	return raw_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char.islower()])
			synonyms.add(synonym) 
	if word in synonyms: synonyms.remove(word)
	return list(synonyms)

def synonym_replacement(words, n, valid_word_list):
	new_words = words.copy()
	random.shuffle(valid_word_list)
	num_replaced = 0
	for random_word in valid_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
		
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def synonym_residue(sentence, words_of_stk, alpha_sr=0.1, num_aug=9):
	valid_words = set(wordnet.words())
	words = get_valid_words(sentence)
	num_words = len(words)

	valid_word_list = []
	for word in words:
		# valid_words.words() are all in lower cases. 
		# If the word is in some STKs, we keep it.
		# If the word contains special charaters like, '#' for (#CITATION_TAG), '@' for 'NDCG@5', it is not a valid word.
		# If the word contains upper letter, it maybe NER, or showing important substructures of the sentence, keep it.
		if (word in valid_words) and (word not in stop_words) and (word not in words_of_stk):
			valid_word_list.append(word)

	valid_word_list = list(set(valid_word_list))

	augmented_sentences = []
	n_sr = max(1, int(alpha_sr*num_words))
	for _ in range(num_aug):
		a_words = synonym_replacement(words, n_sr, valid_word_list)
		augmented_sentences.append(' '.join(a_words))

	return augmented_sentences

def replace_with_alpha(alpha, tmp, a, b):
	if random.random() < alpha:
		return tmp.replace(a, b)
	else: return tmp

	
def choose_with_alpha(alpha, tmp, a, candidates):
	if random.random() < alpha:
		b = random.choice(candidates)
		return tmp.replace(a, b), get_valid_words(b)
	else: return tmp, []

def stk_abstraction(tmp, stks, alpha):
	for stk_name in stks:
		count = 1
		abname = CLASS2AB[stk_name]
		for mention in stks[stk_name]:
			if mention in tmp: # with probability, abstract it to the class name. 
				tmp = replace_with_alpha(alpha, tmp, mention, f'{abname}-{count}')
				count += 1
			elif "'" in mention and mention.replace("'", '"') in tmp:
				tmp = replace_with_alpha(alpha, tmp, mention.replace("'", '"'), f'{abname}-{count}')
				count += 1
			elif '"' in mention and mention.replace('"', "'") in tmp:
				tmp = replace_with_alpha(alpha, tmp, mention.replace('"', "'"), f'{abname}-{count}')
				count += 1
	return tmp, []



def stk_swap(tmp, stks, alpha):
	for stk_name in stks:
		if len(stks[stk_name])<=1:
			continue
		for j, mention in enumerate(stks[stk_name]):
			if mention in tmp: 
				tmp, _ = choose_with_alpha(alpha, tmp, mention, stks[stk_name][:j]+stks[stk_name][j+1:])
			elif "'" in mention and mention.replace("'", '"') in tmp:
				tmp, _ = choose_with_alpha(alpha, tmp, mention.replace("'", '"'),  stks[stk_name][:j]+stks[stk_name][j+1:])
			elif '"' in mention and mention.replace('"', "'") in tmp:
				tmp, _ = choose_with_alpha(alpha, tmp, mention.replace('"', "'"),  stks[stk_name][:j]+stks[stk_name][j+1:])
			
	return tmp, []



def stk_replacement(tmp, stks, alpha, stk_name2mention):
	stk_words = []
	for stk_name in stks:
		for mention in stks[stk_name]:
			update_stk_words = []
			if mention in tmp: 
				tmp, update_stk_words = choose_with_alpha(alpha, tmp, mention, stk_name2mention[stk_name])
			elif "'" in mention and mention.replace("'", '"') in tmp:
				tmp, update_stk_words = choose_with_alpha(alpha, tmp, mention.replace("'", '"'), stk_name2mention[stk_name])
			elif '"' in mention and mention.replace('"', "'") in tmp:
				tmp, update_stk_words = choose_with_alpha(alpha, tmp, mention.replace('"', "'"), stk_name2mention[stk_name])
			stk_words += update_stk_words
	return tmp, stk_words


def get_stk_name2mention(uid2stks, data):
	res = {}
	updated_uid2stk = {}

	for uid in uid2stks:
		if not uid in data.index:
			continue
		updated_uid2stk[uid] = {}

		input_context = data.loc[uid, 'input_context']
		for stk_name in uid2stks[uid]:
			if not stk_name in res: res[stk_name] = []
			if not stk_name in updated_uid2stk[uid]: updated_uid2stk[uid][stk_name] = []

			for mention in uid2stks[uid][stk_name]:
				mention = mention.strip()
				flag = False
				if mention in input_context:
					flag = True					
				elif "'" in mention and mention.replace("'", '"') in input_context:
					mention = mention.replace("'", '"')
					flag = True
				elif '"' in mention and mention.replace('"', "'") in input_context:
					mention = mention.replace('"', "'")
					flag =True
					
				if flag: 
					res[stk_name].append(mention)
					updated_uid2stk[uid][stk_name].append(mention)

	unique_stk_res = {}
	for stk_name in res:
		unique_stk_res[stk_name] = list(set(res[stk_name]))
		print(f"#{stk_name}={len(unique_stk_res[stk_name])}")

		num = [len(updated_uid2stk[uid][stk_name]) if stk_name in updated_uid2stk[uid] else 0 for uid in updated_uid2stk]
		print('Avg.={:.2f}, Std.={:.2f}.'.format(np.mean(num), np.std(num)))

	return unique_stk_res, updated_uid2stk
	


def KP(dataset, num_aug, beta, gamma, all_mode, dump = False):
	random.seed(1)

	mode_name = ''.join(all_mode)
	output_file = f'stk/{dataset}-{mode_name}-{beta}-{gamma}.json'
	if os.path.exists(output_file):
		print("loading from ", output_file)
		with open(output_file, 'r') as fp:
			res = json.load(fp)
		return res

	data = pd.read_csv(f'data/{dataset}_train.txt', sep='\t')
	data = data.set_index('unique_id')

	with open(f'stk/{dataset}_uid2stk.json', 'r') as fp:
		uid2stks = json.load(fp)

	if 'gr' in all_mode:
		with open(f'stk/{dataset}_type2mention.json', 'r') as fp:
			stk_name2mention = json.load(fp)
	else: stk_name2mention = None

	res = {}
	for uid, line in tqdm(data.iterrows()): 
		sentence = line['input_context']

		augmented_sentences = []
		stks = uid2stks[uid]
		
		for _ in range(num_aug):
			tmp = sentence
			mode = all_mode[_%len(all_mode)]

			words_of_stk = []
			for typename in stks:
				for mention in stks[typename]:
					words_of_stk += get_valid_words(mention)
			
			if beta>0:
				if mode == 'ab': augmented_sentence, update_words_of_stk = stk_abstraction(tmp, stks, beta)
				elif mode == 'gr': augmented_sentence, update_words_of_stk = stk_replacement(tmp, stks, beta, stk_name2mention)
				elif mode == 'lr': augmented_sentence, update_words_of_stk = stk_swap(tmp, stks, beta)
				else: raise NotImplementedError
			else:
				augmented_sentence = tmp
				update_words_of_stk = []

			if gamma>=0:
				augmented_sentence = synonym_residue(augmented_sentence, set(words_of_stk+update_words_of_stk), alpha_sr=gamma, num_aug=1)
				augmented_sentences += augmented_sentence
			else: augmented_sentences.append(augmented_sentence)
		res[uid] = augmented_sentences

	if dump:	
		with open(output_file, 'w') as fp:
			json.dump(res, fp)
			print('dump at', output_file)
	print(f"generated KP with stk-{mode_name} for " + dataset + ", num_aug=" + str(num_aug), ", beta=", beta)
	return res


if __name__ == '__main__':
	dataset = 'acl_arc'
	data = pd.read_csv(f'data/{dataset}_train.txt', sep='\t')
	data = data.set_index('unique_id')

	with open(f'stk/{dataset}_stk.json', 'r') as fp:
		uid2stks = json.load(fp)

	stk_name2mention, updated_uid2stk = get_stk_name2mention(uid2stks, data)
	with open(f'stk/{dataset}_type2mention.json', 'w') as fp:
		json.dump(stk_name2mention, fp)

	with open(f'stk/{dataset}_uid2stk.json', 'w') as fp:
		json.dump(updated_uid2stk, fp)
