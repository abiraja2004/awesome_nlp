from __future__ import unicode_literals, print_function, division
import random
from train import trainIters
from evaluate import evaluateRandomly
from data import prepareData
from Decoder import DecoderRNN, AttnDecoderRNN
from Encoder import EncoderRNN, Attentive_Encoder

use_cuda = False

SOS_token = 0
EOS_token = 1


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

teacher_forcing_ratio = 0.5

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

hidden_size = 256
encoder = Attentive_Encoder(input_lang.n_words, hidden_size, 3, use_cuda)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, 1)

teacher_forcing_ratio = 0.5

trainIters(input_lang, output_lang, pairs[:100], encoder, decoder, 75000,
           print_every=50)

evaluateRandomly(input_lang, output_lang, pairs[:100], encoder, decoder)
