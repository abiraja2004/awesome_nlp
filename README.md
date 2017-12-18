# awesome_nlp
## Requirements
#### Python3
- pytorch
- nltk
- matplotlib

#### Python2
- [pke](https://github.com/boudinfl/pke)
- [files2rouge](https://github.com/pltrdy/files2rouge)
-
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
TODO

### RNN
#### Training
TODO
#### Testing
TODO

### RAN
#### Training
TODO
#### Testing
TODO


## Evaluating generated summaries
Using the `files2rouge` package, run:
```
python2 files2rouge.py <gold_standard.txt> <generated_summary.txt>
```
