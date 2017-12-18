# awesome_nlp
## Requirements
#### Python3
- pytorch
- nltk
- matplotlib

#### Python2
- [pke](https://github.com/boudinfl/pke)
- [files2rouge](https://github.com/pltrdy/files2rouge)

## Baselines:
To generate summaries using the First-n and Random baseline, run:
```
python3 test.py --input <input> --words <limit>
```

To generate summaries using the TopicRank keyphrase extraction, run:
```
python2 baselines/pke_baseline.py <input> <limit>
```

## Neural Models
### FFNN
#### Training
TODO
#### Testing
```
python test.py --decoder bms --model <model.pt> --w2i <w2i.pickle> --i2w <i2w.pickle> --nr_docs 10000 --save <summaries.txt> --documents <../dataset/test.article.txt> --summaries <../dataset/test.title.txt>
```

### RNN
#### Training
TODO
#### Testing
```
python test.py --decoder bms --encoder <enc.pt> --rnn_decoder <dec.pt> --w2i <w2i.pickle> --i2w <i2w.pickle> --nr_docs 10000 --save <summaries.txt> --documents <../dataset/test.article.txt> --summaries <../dataset/test.title.txt>
```

### RAN
#### Training
TODO
#### Testing
```
python test.py --decoder bms --encoder <enc.pt> --ran_decoder <dec.pt> --w2i <w2i.pickle> --i2w <i2w.pickle> --nr_docs 10000 --save <summaries.txt> --documents <../dataset/test.article.txt> --summaries <../dataset/test.title.txt>
```

## Evaluating generated summaries
Using the `files2rouge` package, run:
```
python2 files2rouge.py <gold_standard.txt> <generated_summary.txt>
```
