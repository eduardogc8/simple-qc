# simple-qc

Question classification is a component of Question Answering systems for identifying the type of answer a user require. <br>
For instance: <br>
``Who is the queen of the United Kingdom?`` 
demands the name of a **PERSON**, while <br>
``When was the queen of the United Kingdom born?`` 
entails a **DATE**. 

Supervised models have been achieving significant results in this task, however, it commonly dependents on a large amount of labeled data and external resources, e.g., Syntactic Tools or a Lexical Ontology, which makes challenging its employment in low-resource languages.

This project proposes an empirical comparison between Question Classification methods, examining the level of dependence of language resources. 

We propose a manual classification of the current state of art methods in four distinct categories:
* **Low:** The method uses features independent of an external resource, or it can be trained with the own training data — for example, Bag-of-words, TF-IDF and word embedding (trained with the own training data).
* **Medium:** The method uses an unsupervised approach and needs a set of a corpus to train a model — for example, word embedding (trained with externals corpus).
* **High:** The method needs labeled data to train a model, or it uses a knowledge base — for example, a syntactic parser and WordNet.
* **Very High:** The method uses a considerable number of resources specifically for a  language. For example, a set of handcraft rules based on morphological, lexical, and syntactic features from the text.

We use this categorization to perform a comparison in terms of: 
- size of the training set necessary for the learning process; 
- how do they perform in different languages. 

Our experiments revealed that new methods using a low and medium level of dependency could achieve performance equivalent or superior to approaches with a high and very high level of dependency on language resources.

## How to reproduce the results

### Dependencies
- python 3
- jupyter notebook

#### Libraries
- keras
- tensorflow
- sklearn
- pandas
- numpy
- nltk

#### Word embedding files
- Dutch: https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.nl.vec
- English: https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
- Spanish: https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec
- Italian: https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.it.vec
- Portuguese: https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.pt.vec

These files were obtained in MUSE repository (https://github.com/facebookresearch/MUSE)

### Datasets

The directory ``datasets/`` contains the collections used in experiments:
- **UIUC** (https://cogcomp.seas.upenn.edu/Data/QA/QC/)
- **DISEQuA** (https://link.springer.com/chapter/10.1007/978-3-540-30222-3_47)

### Run

The file ``benchmark.ipynb`` contains the code to create the models, load the datasets, and run the experiments. Also, the file disposes of descriptions of the code and how to run it.

The file `features_extract.ipynb` contains the code to extract features from question text using the Spacy syntactic tool (https://spacy.io/). Once it has already been executed, it not necessary to rerun it.

The file `plots.ipynb` is used to view the results.
