# Q^2

Code and data accompanying the paper "Q^2: Evaluating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering".
Q^2 is a reference-less metric that aims to evaluate the factual consistency of knowledge-grounded dialogue systems.
Our approach is based on automatic question generation and question answering.

## Datasets

The datasets are stored in the `data` folder. 
The folder contains two files - consistent and inconsistent responses - for each of the two systems. Each file contains 150 responses.
In addition, we add the cross anotation file, used in our validation experiments.

The datasets are stored as pandas dataframes in csv files. Loading them should be as simple as:

```
In [1]: import pandas as pd

In [2]: data = pd.read_csv("data/dodeca_consistent.csv")

In [3]: data.head()
Out[3]:
   episode_idx  round  ...                                           response                                          knowledge                                               gold
0            0      0  ...  i love gardening as well . it is considered to...  Gardening is considered by many people to be a...  I live on a farm, we garden all year long, it ...
1            1      2  ...      it aired in the us , canada and latin america  He was the creator and host of "The Joy of Pai...           The show aired from 1983 to 1994 on PBS.
2            4      1  ...  well , there are three categories of finance :...  Finance can be broken into three sub-categorie...  Great! There are three categories. The public,...

```

The key columns are:
- `episode_idx`: The index of the conversation in the WoW validation data.
- `round`: The turn in the conversation.
- `response`: The model's response for the given turn.
- `knowledge`: The knowledge given to the model for the given turn.
- `gold`: The gold-standard (human) response for the given turn.

## Scripts to reproduce papers results

### Prerequisites
* python 3.7
* numpy==1.19.2
* pandas==1.1.3
* bert-score==0.3.7
* spacy==2.3.2
* torch==1.6.0
* transformers==3.2.0

To compare against baselines and run validation experiments, you'll additionaly need:
* scikit-learn==0.20.2
* scipy==1.1.0
* sacrebleu==1.4.14

For the coreference-resolution preprocessing:
* allennlp==1.0.0
* pytorch-truecaser, which can be installed from https://github.com/mayhewsw/pytorch-truecaser


### Usage
To run Q^2, run `pipeline/run_pipeline.py` and specify the parameters. 
For example:
```
python pipeline/run_pipeline.py \
      --infile data/dodeca_inconsistent.csv \
      --gen_method beam \
      --q_per_cand single \
      --personal remove \
      --outfile dodeca_inconsistent_beam_single_remove.csv
```

Validation experiments:
For score robustness, first run 
```
python baselines.py \
      --infile dodeca_inconsistent_q2.csv \
      --outfile dodeca_inconsistent_q2_baselines.csv
```

When the infile parametr is a csv file containing all the columns that exist in the data files, with an additional column of Q^2 scores. 
Such file is obtained using the previous script, pipeline/run_pipeline.py.
Run this script for each of the 4 data files, dodeca/memenet, consistent and inconsistent.

Then, run
```
python score_robustness.py \
      --incons_dodeca_f dodeca_inconsistent_q2.csv \
      --cons_dodeca_f dodeca_consistent_q2.csv\
	  --incons_memnet_f memnet_inconsistent_q2.csv \
      --cons_memnet_f memnet_consistent_q2.csv
```

For system-level evalution, first run 
```
python pipeline/prepare_files.py \
      --infile cross_anotation_q2.csv \
      --outfile cross_anotation_q2_baselines.csv
```

When the infile parametr is a csv file obtained running pipeline/run_pipeline.py on the `cross_anotation.csv` file.
Then, run
```
python sys_level.py \
      --infile cross_anotation_q2_baselines.csv
```


