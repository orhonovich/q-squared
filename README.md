# Q²

Code and data accompanying the paper "Q²: Evaluating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering".
Q² is a reference-free metric that aims to evaluate the factual consistency of knowledge-grounded dialogue systems.
Our approach is based on automatic question generation and question answering.

## Datasets

The datasets are available in the `third_party/data` folder.
They are based on a subset of the [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) dataset.
Specifically, we ran inference using two dialogue systems and manually annotated each response for factual consistency. 
The data is available in four files - consistent and inconsistent responses - for each of the two systems. Each file contains 150 responses.
In addition, we add the cross anotation file, used in our validation experiments.

The datasets are stored as pandas dataframes in csv files. Loading them should be as simple as:

```
In [1]: import pandas as pd

In [2]: data = pd.read_csv("third_party/data/dodeca_consistent.csv")

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

To compare against baselines and run validation experiments, you'll additionally need:
* scikit-learn==0.20.2
* scipy==1.1.0
* sacrebleu==1.4.14

For the NLI-based comparison:
* allennlp==1.0.0
* allennlp-models==1.0.0



### Usage
To run Q^2, first run `pipeline/run_pipeline.py` and specify the parameters. 
Use the save_steps flag, which will later enable measuring answer similarity using an NLI system.
For example:
```
python pipeline/run_pipeline.py \
      --infile third_party/data/dodeca_inconsistent.csv \
      --gen_method beam \
      --q_per_cand single \
      --personal remove \
      --outfile dodeca_inconsistent_out \
      --save_steps
```

Then, run `nli_spans_comparison.py` with the steps file generated at the previous step (in the example above, 
`dodeca_inconsistent_out.steps.csv`). 
For example:
```
python nli_spans_comparison.py \
      --infile dodeca_inconsistent_out.steps.csv \
      --outfile dodeca_inconsistent_scores.csv
```


### Meta-evaluation experiments

#### Response-Level Precision and Recall

The response-level evaluation includes measuring the Precision/Recall tradeoff of inconsistency and consistency 
detection at a single-example level, using various thresholds.

To plot the Precision-Recall curve vs. various thresholds, run `precision_recall.py` and specify the parameters.
```
python precision_recall.py \
      --incons_dodeca_f dodeca_inconsistent_scores.csv \
      --cons_dodeca_f dodeca_consistent_scores.csv \
	  --incons_memnet_f memnet_inconsistent_scores.csv \
      --cons_memnet_f memnet_consistent_scores.csv \
      --metrics_names 'Q2' 'Q2_no_nli'
```
Each input file should be obtained by running `pipeline/run_pipeline.py` followed by `nli_spans_comparison.py`, as 
explained under Usage.

The output files will include two plots for each input metric: grounded and ungrounded Precision and Recall vs. various 
thresholds. If more than one metric was provided as an input, the output will include two additional plots, comparing 
the grounded and ungrounded Precision-Recall trade-off for all input metrics. Other than the plots, the accuracy given a
specific threshold, as well as the grounded and ungrounded Precision and Recall, will be printed.

metrics_names should be one or more space-separated names of the tested metrics.
For the specific threshold computation, use the `thresholds` flag, which should be one or more space-separated values of 
thresholds, one for each specified metric name selected. If thresholds weren't specified, the computation will use a 
threshold of 0.5.
To add baseline methods to the Precision-Recall computation, specify the `add_baselines` flag.

##### Comparing to new metrics
To compare new metrics to q-squared, add a column containing the new metric's scores for each of the above csv files,
and add the name of this column to the names passed in the `metrics_names` flag. Note that scores should be normalized
to [0,1].

#### System-Level Evaluation

For system-level evaluation, first run `pipeline/prep_sys_experiment.py` and specify the parameters.
The `infile` should be the file containing the extended annotations, for both the dodeca and memnet systems.
```
python pipeline/prep_sys_experiment.py \
      --infile third_party/data/cross_annotation.csv \
      --outfile cross_annotation_out
```

This will create two output files - one for each system: `cross_annotation_out_dodeca.csv`, and  
`cross_annotation_out_memnet.csv`
Then, run `nli_spans_comparison.py` for each of the two files and use the `for_systems_simulation` flag:
For example:
```
python nli_spans_comparison.py \
      --infile cross_annotation_out_dodeca.csv \
      --outfile cross_annotation_dodeca_scores.csv \
      --for_systems_simulation
```

Finally, run `system_level.py` with the two files generated at the previous step.
```
python system_level.py --dodeca_path cross_annotation_dodeca_scores.csv --memnet_path cross_annotation_memnet_scores.csv
```

### Cite
```
@inproceedings{honovich-etal-2021-evaluating,
    title = "Q²: Evaluating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering",
    author = "Honovich, Or  and
      Choshen, Leshem  and
      Aharoni, Roee  and
      Neeman, Ella  and
      Szpektor, Idan  and
      Abend, Omri",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2104.08202",
}
```