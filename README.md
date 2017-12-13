# awesome_nlp

## Requirements
- [sumy](https://pypi.python.org/pypi/sumy/0.7.0)

## New Plan of attack
- Implement two Facebook papers using `pytorch`:
  - A neural attention model for abstractive sentence summarization
  - Abstractive sentence summarization with attentive recurrent neural networks
- Compare results
- Contribute to abstractive summarization research with some ground-braking new stuff

## Implementing 'A neural attention model for abstractive sentence summarization' by Verna
Disclaimer: I am bad at writing readmes.
- The general flow of initializing a model and training the model happens in main.py. It does not work with batches YET. <-- Working on it.
- In data.py some useful classes are implemented for data representation: Dictionary (with word2idx and idx2word), Corpus (train, test, validation), Text (with its own Dictionary, words and sentences) and Collection (contains a list of documents and a list of corresponding summaries).
- All the functions that are used for adapting the data but were not yet embedded in the classes from data.py are now in utils.py.
- NPLM_Summarizer in NPLM.py is the general neural language model as represented in Figure 3a. NPLM_Summarizer uses an Encoder object to calculate the encoding within the forward function.
- Encoder.py currently contains two encoders, BOW_Encoder and Attention_Based_Encoder.
- Decoder.py contains both a Greedy and a Beam Search decoder that need a (trained) model and input documents to generate summaries. Beam Search decoder needs to be initialized with some parameters, such as beam size. If you want to check the hypothesis generation process, use the verbose mode.
- The way main.py works now, is that it estimates summaries after every epoch (with a greedy decoder) and saves them in 'summaries.txt', so that you can keep track of the summaries that are generated while the algorithm is running.
- Adam seems to work much faster than SGD.
