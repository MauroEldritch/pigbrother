# pigbrother

## Introduction
**pigbrother** uses neural networks and Markov chains to generate political propaganda based on any media, with different affiliations of the political spectrum, and in any language. Also, it identifies clickbait and possible fake news. 

## Build & Usage
### 1. Install
- Requirements:
    - Python 3+ 
    - feedparser
    - tensorflow
    - textgenrnn
    - markovify
    - keras
    - gensim

Having Python 3 installed, just run: `pip3 install -r core/requirements.txt`

### 2. Feeding RSS sources
Create csv files in the input folder, using the following format:

`rss_url,affiliation`

Where affiliation can be `left`, `right` or `garbage`.

- Choosing left will store the headings on the left-wing collection.
- Choosing right will do so for the right-wing collection.
- Choosing garbage will collect confirmed clickbaits only, into the garbage collection.

Examples:
```
cat pagina12.csv
left-wing-media.com/rss,left

cat cabildo.csv
right-wing-news.com/rss_feed,right

cat mysupernewssitenotfakeatall.csv
awesome-clickbait.com/rss.xml,garbage
```

### 3. Executing pigbrother (Syntax)

- **Basic Syntax:**

`./pigbrother.py [collect | train {affiliation, iterations, terms} | generate_full {model_name} | generate_light {affiliation} | generate_custom {affiliation, start_word} | test | purge | help]`

- *help* / *[no argument]*: Displays a help block. Pretty much the same information as this document.
- *collect*: Starts fetching and parsing data from the RSS sources described in the `input` folder.
- *purge*: Truncates all the `output` files. Sometimes learning over older news mess up with the desired output.
- *train [affiliation] [iterations] [terms]*: Trains a model based on the output files for the given affiliation. For example, using `train right 100 Trump Putin Maduro` will read `output/rightwingnews.csv` for training a model using 100 iterations and focusing on the three politicians names given. The three terms are mandatory.
- *generate_full [model_name]*: Attempts to generate propaganda using a neural network model previously trained. This is the most **experimental** and **challenging** experience, as neural networks output tend to fail and be messy at first, but offers the possibility of *improving* over the time, with more/better training, or different tunning.
- *generate_light [affiliation]*: Attempts to generate propaganda with a given affiliation using Markov chains, without the need of a previously trained model. This is the fastest way and can provide acceptable results on the very first attempts, but bear in mind that this way is a "one-shot", and it's not going to "improve" over time.
- *generate_custom [affiliation] [start_word]*: Generates specially crafted headlines. For example, running `generate_custom left Macri` will output left-wing oriented headlines starting with the argentine president's surname.
- *test*: Launches the Replicant Test, in order to measure pigbrother's semantic skills with the collected data (See `Section 5: Testing (Or "The Replicant Test)`).

### 4. Understanding folders structure
- **core:** Internal files, like `clickbaitwords.csv` dictionary, `requirements.txt`, and `README.md` media. Aside from the clickbait dictionary and the configuration file `config.py`, the rest is not meant to be modified unless you know what you are doing.
- **input:** Your CSV files stating an RSS source and its affiliation will reside here (See `Section 2: Feeding RSS Sources`).
- **output:** Your CSV files containing processed headlines (separated by affiliation) will be stored here.
- **models:** After you train a model, it will be stored here for later use. The naming convention is `model_affiliation_number-of-lines_x_number-of-epochs.h5` for the model, and the same but with `.keywords` extension for the keywords file. Keywords are the three mandatory terms you have to provide to pigbrother in order to train a model.
- **docs:**: You can find a copy of our whitepaper here.

### 5. Testing (Or "The Replicant Test")
Once you've collected enough data (around 100 headlines), you should try the `test` switch, which will launch an interactive testing menu.
The Replicant Test will try to generate fake headlines using Markov chains (with **pigbrother**'s `generate_light` module), then you will be prompted to choose one of them to be hidden among other real headlines. You may ask a friend to try and guess the fake headline interactively, helping you to see if your sources are useful and reliable, and if you're on the right track.

### 6. Examples (TL;DR)
```
#Collect data from RSS Feeds
./pigbrother.py collect

#Train a model
./pigbrother.py train left 5 macri vidal larreta

#Generate headlines using a trained model
./pigbrother.py generate_full model_left_1766_x_5.h5

#Generate headlines using Markov Chains
./pigbrother.py generate_light right

#The Replicant Test
./pigbrother.py test
```

## Diagram

![pigbrother.py functional diagram](core/img/pigbrother.png?raw=true "pigbrother.py functions")

## Presentations
### As Pigbrother
|#| Date | Conference |  Link to Video | Link to Slides |
|---|---|---|---|---|
|1|2020|GrayHat| https://www.youtube.com/watch?v=Ger2u59bqWE | https://docs.google.com/presentation/d/1R72cwAZinC4cgbU_TyxZRZKYpfz4EroKaQ8Az4aiAvA/edit?usp=sharing |

### As VKG
|#| Date | Conference |  Link to Video | Link to Slides |
|---|---|---|---|---|
|1|2021|P0SCon Iran| - | https://drive.google.com/file/d/11jSrcWHsQEGgQVbmHlxG9D0S1DN0o4Pr/view?usp=sharing |
|2|2021|Machine Learning Utah| - | https://docs.google.com/presentation/d/1-AEVqTtDlrwQ4Ekj7IMycSzEwC3uZUYuxP0rEH1Wd5g/edit?usp=sharing |

## Credits
**pigbrother** was created by @[flordiaz9](https://github.com/flordiaz9) and @[mauroeldritch](https://github.com/mauroeldritch) in 2019. 