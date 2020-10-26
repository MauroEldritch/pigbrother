#Pigbrother Configuration File
#Critical and trivial settings coexist in this file, modify only if you know what you are doing.

#Neural Networks Settings (Critical, handle with care)
#Sentence Max Length
max_sentence_len = 40
#Word Model Settings
word_model_size = 10
word_model_window = 4
word_model_min_count = 1
word_model_iter = 100
#Sample Temperature
sample_temperature = 0.7
#Activation
#Warning: Modifying this value may require to heavily alter the codebase of pigbrother.  
activation = 'softmax'

#Trainer-Exclusive Settings
#Optimizer
trainer_optimizer = 'adam'
#Loss
trainer_loss = 'sparse_categorical_crossentropy'
batch_size = 128

#Generator-Exclusive Settings
#Optimizer
generator_optimizer = 'adam'
#Loss
generator_loss = 'categorical_crossentropy'
#Output Count
generator_output_count = 10

#Markovify Settings (Safe to modify)
markov_chain_length = 100

#Banners (Safe to modify)
banner = """
      ,,__
    c''   )~	pigbrother V0.1
      ''''     		|_ mauroeldritch / flordiaz9
"""
replicant_banner = """
 _________________
| |     ,,__    |o|
| |   c''   )~  | |
| |     ''''    | |  Replicant Test V0.1
| |_____________| |		|_ mauroeldritch / flordiaz9
|     _______     |
|    |       |    |
| RT |       |    |
|____|_______|____|
"""