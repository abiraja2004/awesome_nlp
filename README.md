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
```
python main.py  --nr_docs 1000 --batch_size 5 --lr 0.05 --epochs 5 --documents <../dataset/valid.article.txt> --summaries <../dataset/valid.title.txt> --em_file <glove.6B.200d.txt> --nhid 200 --context_size 5 --encoder att --save <model/>
```
The model trained and dictionaries needed for testing will then be saved in the folder 'model'.
To use the BoW encoder, replace 'att' with 'bow'.

#### Testing
```
python test.py --decoder bms --model <model.pt> --w2i <w2i.pickle> --i2w <i2w.pickle> --nr_docs 10000 --save <summaries.txt> --documents <../dataset/test.article.txt> --summaries <../dataset/test.title.txt>
```
To use the greedy decoder, replace 'bms' with 'grd'.
In the folder 'model', a very small example model is provided, highly overfitted on a small part of the data to illustrate the workings of the model.

### RNN
#### Training
```
python main.py  --nr_docs 100 --batch_size 5 --lr 0.005 --epochs 5 --q 5 --documents <train.article.txt> --summaries <train.title.txt>
```
#### Testing
```
python test.py --decoder bms --encoder <enc.pt> --rnn_decoder <dec.pt> --w2i <w2i.pickle> --i2w <i2w.pickle> --nr_docs 10000 --save <summaries.txt> --documents <../dataset/test.article.txt> --summaries <../dataset/test.title.txt>
```

To use the greedy decoder, replace 'bms' with 'grd'.

### RAN
#### Training
```
python main.py  --nr_docs 100 --batch_size 5 --lr 0.005 --epochs 5 --q 5 --documents <train.article.txt> --summaries <train.title.txt>
```


#### Testing
```
python test.py --decoder bms --encoder <enc.pt> --ran_decoder <dec.pt> --w2i <w2i.pickle> --i2w <i2w.pickle> --nr_docs 10000 --save <summaries.txt> --documents <../dataset/test.article.txt> --summaries <../dataset/test.title.txt>
```

To use the greedy decoder, replace 'bms' with 'grd'.

## Evaluating generated summaries
Using the `files2rouge` package, run:
```
python2 files2rouge.py <gold_standard.txt> <generated_summary.txt>
```
