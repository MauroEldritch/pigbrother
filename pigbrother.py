#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#pigbrother - Mauro Cáseres (@mauroeldritch) | Florencia Diaz (@FlorDiaz_) - 2019

#Internal configuration
from core import config
#Common imports
import glob, os, csv, re, sys, random
from datetime import datetime
from random import randint
#RSS feeds parser
import feedparser
#Neural network trainer and dependencies, for train & generate_full modes
import numpy as np
import gensim, string, warnings
#Markov chain generator, for test, generate_light & generate_custom modes
import markovify
os.system('clear')
#Global keywords
keywords_clickbait = []
train_words = []
debug_info = 1

#Print help
def showhelp():
	help_message = """
Usage: ./pigbrother.py 
	[collect
	|train {left, right, garbage} {iterations} {train_word_a} {train_term_b} {train_term_c}
	|generate_full {model_name}
	|generate_light {left, right, garbage}
	|generate_custom {left, right, garbage} {start_word}
	|test
	|purge
	|help]`

- help: Displays this message.
- collect: Fetch and parse RSS sources in 'input'.
- purge: Truncate output files.
- train [affiliation] [iterations] [terms]: Trains a model based on given affiliation news. 
- generate_full [model_name]: Generates propaganda using a neural network model previously trained.
- generate_light [affiliation]: Generates propaganda with a given affiliation using Markov chains.
- generate_custom [affiliation] [start_word]: Generates headlines using specific affiliation and starting with a given word.
- test: Launches an interactive test for the user to detect a fake headline among other real ones.


Refer to the README.md file to get detailed information.
"""
	print(help_message)

#Determine which file to read from, given the desired affiliation
def pick_news_file(affiliation):
	if str(affiliation) == "left":
		news_file = "output/leftwingnews.csv"
	elif str(affiliation) == "right":
		news_file = "output/rightwingnews.csv"
	elif str(affiliation) == "garbage":
		news_file = "output/garbagenews.csv"
	return news_file

#Check if title seems like a clickbait, 2 = high chance, 1 = likely
def clickbait_rating(news):
	bait_rating = 0
	if news.lower().startswith(tuple(keywords_clickbait)) or news.lower()[0].isdigit():
		bait_rating = 2
	elif any(word in news.lower() for word in keywords_clickbait):
		bait_rating = 1
	return bait_rating
	
#Output news to the corresponding file.
def generate_output(media_title,media_affiliation,clickbait_score):
	#If garbage and confirmed/suspected clickbait, store in garbage. If garbage but not clickbait, ignore.
	new_entry = [[media_title]]
	if media_affiliation == "garbage" and clickbait_score > 0:
		csv_file = 'output/garbagenews.csv'
	#If right, store in right-wing
	elif media_affiliation == "right":
		csv_file = 'output/rightwingnews.csv'
	#If left, store in left-wing
	elif media_affiliation == "left":
		csv_file = 'output/leftwingnews.csv'
	try:
		with open(csv_file, 'a') as f:
			writer = csv.writer(f)
			for row in new_entry:
				writer.writerow(row)
	except:
		pass

#Clean output files, removing special characters and duplicated lines.
def clean_output_files():
	output_files = ['output/leftwingnews.csv','output/rightwingnews.csv','output/garbagenews.csv']
	for output_file in output_files:
		lines_seen = set()
		outfile = open('core/dict/tempstorage.csv', "w")
		for line in open(output_file, "r"):
			if line not in lines_seen:
				line_clean = re.sub('[“”"]', '', line)
				outfile.write(line_clean)
				lines_seen.add(line)
		outfile.close()
		os.remove(output_file)
		os.rename("core/dict/tempstorage.csv", output_file)
	print ("[*] All output files were cleaned successfully.")

#Collect information from RSS sources and create CSV outputs.
def collect():
	#Load keywords into arrays
	global keywords_clickbait
	with open('core/dict/clickbaitwords.csv') as f:
		for line in f:
			keywords_clickbait = line.split(',')
	datasets = []	
	datafiles = glob.glob('input/*.csv')
	for file in datafiles:
		with open(file, 'rt') as f:
			reader = csv.reader(f)
			for row in reader:
				datasets.append(row)
	#Dataset processing
	left_news_count = 0
	right_news_count = 0
	garbage_news_count = 0
	baits_count = 0
	probably_baits_count = 0
	for dataset in datasets:
		if debug_info == 1:
			print ("[URL: " + dataset[0] +"]")
		d = feedparser.parse(dataset[0])
		media_name = d.feed.title
		media_affiliation = dataset[1]
		print ("[*] " + media_name + " (" + str(len(d['entries'])) + " entries) - [" + media_affiliation + "]")
		for entry in d.entries:
			media_title = entry.title
			clickbait_score = clickbait_rating(entry.title)
			if clickbait_score == 1:
				status = "?"
				probably_baits_count += 1
			elif clickbait_score == 2:
				status = "!"
				baits_count += 1
			else:
				status = "-"
			if media_affiliation == "right":
				wing = "R"
				right_news_count += 1
			elif media_affiliation == "left":
				wing = "L"
				left_news_count += 1
			elif media_affiliation == "garbage":
				wing = "X"
				garbage_news_count += 1	
			print ("["+ status +"][" + wing + "] " + media_title)
			#Write output files
			generate_output(media_title,media_affiliation,clickbait_score)
		print ("\n")
	#End dataset processing with a line break
	clean_output_files()
	print ("[*] " + str(right_news_count) + " news classified under 'right-wing'.")
	print ("[*] " + str(left_news_count) + " news classified under 'left-wing'.")
	print ("[*] " + str(garbage_news_count) + " news classified under 'garbage'.")
	print ("[*] " + str(baits_count) + " news are confirmed clickbaits.")
	print ("[*] " + str(probably_baits_count) + " news are probably clickbaits.")

#Purge output files
def purge():
	output_files = ['output/leftwingnews.csv','output/rightwingnews.csv','output/garbagenews.csv']
	for output_file in output_files:
		with open(output_file, 'w'): pass
	print("[*] Files truncated successfully.")

#Create the neural network.
def build_neural_network(affiliation, mode, iterations=0,model_file="none",key_file="none"):
	#iterations is for train mode only; model_file and key_file are for generate_full mode only.
	#Both sides are properly neutralized when not needed.
	#You can tune the Neural Network layout in the core/config.py file.
	from keras.callbacks import LambdaCallback
	from keras.layers.recurrent import LSTM
	from keras.layers.embeddings import Embedding
	from keras.layers import Dense, Activation
	from keras.models import Sequential
	from keras.utils.data_utils import get_file
	
	#Functions and settings for both, training and generation
	def sample(preds, temperature=1.0):
		if temperature <= 0:
			return np.argmax(preds)
		preds = np.asarray(preds).astype('float64')
		preds = np.log(preds) / temperature
		exp_preds = np.exp(preds)
		preds = exp_preds / np.sum(exp_preds)
		probas = np.random.multinomial(1, preds, 1)
		return np.argmax(probas)
	def generate_next(text, num_generated=10):
		word_idxs = [word2idx(word) for word in text.lower().split()]
		for i in range(num_generated):
			prediction = model.predict(x=np.array(word_idxs))
			idx = sample(prediction[-1], temperature=config.sample_temperature)
			word_idxs.append(idx)
		return ' '.join(idx2word(idx) for idx in word_idxs)
	def on_epoch_end(epoch, _):
		print("[*] Similar words:")
		for word in train_words:
			sample = generate_next(word)
			print('%s... -> %s' % (word, sample))
	def word2idx(word):
			return word_model.wv.vocab[word].index
	def idx2word(idx):
			return word_model.wv.index2word[idx]

	#Supress warnings
	warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
	news_file = pick_news_file(affiliation)

	#Create sentences collection
	max_sentence_len = config.max_sentence_len
	with open(news_file,encoding='utf-8') as news_file:
		docs = news_file.readlines()
	sentences = [[word for word in doc.lower().translate(str.maketrans('','',string.punctuation)).split()[:max_sentence_len]] for doc in docs]
	word_model = gensim.models.Word2Vec(sentences, size=config.word_model_size, window=config.word_model_window, min_count=config.word_model_min_count, iter=config.word_model_iter)
	pretrained_weights = word_model.wv.vectors
	vocab_size, emdedding_size = pretrained_weights.shape
	
	#Templating the model
	model = Sequential()
	model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
	model.add(LSTM(units=emdedding_size))
	model.add(Dense(units=vocab_size))
	model.add(Activation(config.activation))
	model.compile(optimizer=config.trainer_optimizer, loss=config.trainer_loss)    

	#Mode-specific procedures
	if str(mode) == "train":
		print("[*] Training affiliation: " + str(affiliation))	
		print("[*] Key terms: " + str(train_words))
		print("[*] Training model with " + str(len(sentences)) + " sentences and " + str(iterations) + " iterations.")
		print("[*] Embedding shape: " + str(pretrained_weights.shape))
		train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
		train_y = np.zeros([len(sentences)], dtype=np.int32)
		for i, sentence in enumerate(sentences):
			for t, word in enumerate(sentence[:-1]):
				train_x[i, t] = word2idx(word)
				train_y[i] = word2idx(sentence[-1])
		print("[*] train_x shape:" + str(train_x.shape))
		print("[*] train_y shape:" + str(train_y.shape))
		print("[*] Checking similar words:")
		for word in train_words:
			most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
			print('  %s -> %s' % (word, most_similar))
		print('[*] Training LSTM...')
		model.fit(train_x, train_y,
			batch_size=config.batch_size,
			epochs=int(iterations),
			callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
		model_name = "models/model_" + affiliation + "_"+ str(len(sentences)) + "_x_" + str(iterations)
		model_to_save = model_name + ".h5"
		keywords_to_save = model_name + ".keywords"
		model.save_weights(model_to_save)
		with open(keywords_to_save, 'w') as f:
			for keyword in train_words:
				f.write("%s\n" % keyword)
		print("\n[*] Model saved to: " + model_to_save)
		print("[*] Keywords saved to: " + keywords_to_save)
	elif str(mode) == "generate":
		model.load_weights("models/" + str(model_file))
		model.compile(loss=config.generator_loss, optimizer=config.generator_optimizer)
		print("[*] Model loaded successfully.")
		with open("models/" + key_file) as f:
			for line in f:
				print("\n[*] Generating content with the keyword: " + line)
				for i in range(config.generator_output_count):
					print("\t" + str(i + 1) + ") "+ generate_next(str(line)))
		print("\n[?] If your results are not satisfactory, you can keep training your model further.")

#Train a given model
def train(affiliation, iterations, train_term_a, train_term_b, train_term_c):
	global train_words
	train_words.extend((train_term_a,train_term_b,train_term_c))
	build_neural_network(affiliation, mode="train", iterations=iterations)

#Generate headlines based on a previously trained model
def generate_full(model_file):
	keywords_file = str(model_file).split(".")[0] + ".keywords"
	affiliation = str(model_file).split("_")[1]
	print("[*] Searching for model file: " + str(model_file))
	if os.path.isfile("models/" + str(model_file)):
		print("[*] Model found.\n[*] Searching for keywords file: " + keywords_file)
		if os.path.isfile("models/" + str(keywords_file)):
			print("[*] Keywords file found.")
			build_neural_network(affiliation,mode="generate",model_file=model_file,key_file=keywords_file)
		else:
			print("[!] Keywords file missing.")
			exit()
	else:
		print("[!] Model file missing.")

#Generate headlines following defined rules
def generate_custom(affiliation, start_word):
	print("[*] Generating 10 " + str(affiliation) + " oriented headlines starting with "+ str(start_word) +", using Markov chains.")
	news_file = pick_news_file(affiliation)
	with open(news_file) as f:
		text = f.read()
		text_model = markovify.NewlineText(text)
	for i in range(10):
		print(str(i+1) +") " +text_model.make_sentence_with_start(str(start_word), "*"))

#Generate headlines using Markov chains
def generate_light(affiliation):
	print("[*] Generating 10 " + str(affiliation) + " oriented headlines using Markov chains.")
	news_file = pick_news_file(affiliation)
	try:
		with open(news_file) as f:
			text = f.read()
			text_model = markovify.NewlineText(text)
		for i in range(10):
			print(str(i+1) +") " +text_model.make_short_sentence(config.markov_chain_length))
	except:
		print("[!] Error while trying to generate fake headlines. Please check that your files contain at least 10 entries before proceeding.")

#The Replicant Test
def replicant():
	#Step One: Choosing political affiliation
	os.system("clear")
	print (config.replicant_banner)
	print ("[?] Select affiliation:\n\t1) Left\n\t2) Right\n\t3) Garbage\n\tx) Quit\n")
	user_option = input("Your choice: ")
	csv_file = ""
	fake_news = []
	print ("\nFake news:")
	if str(user_option) in "1 2 3 x":
		if user_option == "1":
			csv_file = 'output/leftwingnews.csv'
		elif user_option == "2":
			csv_file = 'output/rightwingnews.csv'
		elif user_option == "3":
			csv_file = 'output/garbagenews.csv'
		elif user_option == "x":
			print("[*] Test aborted. Goodbye.")
			exit()
		with open(csv_file) as f:
			text = f.read()
			text_model = markovify.NewlineText(text)
		for i in range(10):
			fake = text_model.make_short_sentence(140)
			fake_news.append(str(fake) +"\n")
			print(str(i) +") "+str(fake))
			#Step Two: Choosing a suitable fake headline
		user_news = input("\n[?] Select a fake news headline [0-9]: ")
		if str(user_news) in "0 1 2 3 4 5 6 7 8 9":
			os.system("clear")
			print (config.replicant_banner)
			test_advice = """READ THIS!

This fake headline will be listed among other 9 real headlines after you hit ENTER on the next prompt.
Get someone else to read the newly listed headlines and attempt to identify the fake news generated on this step.

Good luck trying to beat the machine.
			"""
			print (test_advice)
			test_start = input("[*] Press ENTER when you're ready to start the test... ")
			#Step Three: Guest may attempt to guess the fake headline among other real ones.
			os.system("clear")
			print(config.replicant_banner)
			news_list = []
			news_list.append(fake_news[int(user_news)])
			file = open(csv_file)
			lines = file.readlines()
			lineno = len(lines)
			lines_to_use = [randint(1,lineno) for p in range (0,9)]
			for i in lines_to_use:
				news_list.append(lines[i])
			random.shuffle(news_list)
			for i in range(10):
				print(str(i) +") " + news_list[i] )
			fake_guess = input("Which one is fake? [0-9]: ")
			if str(fake_guess) in "0 1 2 3 4 5 6 7 8 9":
				if str(news_list[int(fake_guess)]) == str(fake_news[int(user_news)]):
					print ("\n[*] Right answer! You spotted the fake headline.\n\nThanks for trying the Replicant Test!\n")
				else:
					print ("\n[!] Wrong answer! The fake headline was: " + str(fake_news[int(user_news)]) +"\n\nThanks for trying the Replicant Test!\n")
			else:
				print("[!] Must specify a number between 0-9. Aborting.")
		else:
			print("[!] Must specify a number between 0-9. Aborting.")
	else:
		print("[!] Error. Aborting.")

#Entrypoint - Main Menu
def main():
	print(config.banner)
	if len(sys.argv) <= 1:
		showhelp()
	elif str(sys.argv[1]) == "collect":
		collect()
	elif str(sys.argv[1]) == "purge":
		purge()
	elif str(sys.argv[1]) == "train" and len(sys.argv) == 7:
		if str(sys.argv[2]) in "left right garbage":
			train(str(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
		else:
			print ("[!] Argument must be 'left', 'right', or 'garbage'. Aborting.")
	elif str(sys.argv[1]) == "generate_custom" and len(sys.argv) == 4:
		generate_custom(str(sys.argv[2]),str(sys.argv[3]))
	elif str(sys.argv[1]) == "generate_full" and len(sys.argv) == 3:
		generate_full(str(sys.argv[2]))
	elif str(sys.argv[1]) == "generate_light" and len(sys.argv) == 3:
		if str(sys.argv[2]) in "left right garbage":
			generate_light(str(sys.argv[2]))
		else:
			print ("[!] Argument must be 'left', 'right', or 'garbage'. Aborting.")
	elif str(sys.argv[1]) == "test":
		replicant()
	else:
		showhelp()

main()