# Tutorial 3: Advanced

The first two tutorials introduced how to load various text datasets and use the framework with default parameters.
This tutorial expands on that by explaining how to select a different transferability metric and rank layers of a single LM using the TransformerRanker.

### Transferability Metrics 

Transferability metrics help estimate how well a model can use knowledge from one task to perform another.
For pre-trained language models (LMs), this involves estimating how well the extracted embeddings are suited for a downstream dataset.
In TransformerRanker, we embed a dataset with different LMs and compare how well embeddings match the task labels.
To score the embeddings, we use one of the three metrics:

- __k-Nearest Neighbors (k-NN)__: Uses distance metrics to measure how closely embeddings from the same class are clustered. We calculate pairwise distance matrix and exclude self-distances in the top _k_ search. [See k-NN code](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/nearestneighbors.py).
- __H-Score__: Measures the feature-wise variance between embeddings of different classes. High variance with low feature redundancy suggests strong transferability. [See H-Score code](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/hscore.py).
- __LogME__: Calculates the log marginal likelihood of a linear model fitted to embeddings. It optimizes two parameters, _alpha_ and _beta_, to adjust the model's regularization and the precision of the prior distribution. [See LogME code](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/logme.py).

We use two state-of-the-art metrics: LogME and an improved H-Score with shrinkage-based adjustments to the covariance matrix calculation.
To use LogME, set the `estimator` parameter when running the ranker:

```python3
result = ranker.run(language_models, estimator="logme")
```

To use embeddings from different layers of a model, set the `layer_aggregator` parameter.

```python3
result = ranker.run(language_models, estimator="logme", estimator="bestlayer")
```

This configuration scores all layers of a language model and selects the one with the highest transferability score.
Models are then ranked based on their best-performing layers for the dataset.

### Layer Ranking

The `bestlayer` option can be used to rank layers of a single LM.
Here's an example using the large encoder model deberta-v2-xxlarge (1.5 billion parameters and 48 layers) with the CoNLL-03 Named Entity Recognition (NER) dataset:

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the CoNLL dataset
conll = load_dataset('conll2003')

# Use a single language model
language_model = ['microsoft/deberta-v2-xxlarge']

# Initialize the ranker and downsample the dataset
ranker = TransformerRanker(dataset=conll, dataset_downsample=0.2)

# Run it using the 'bestlayer' option
result = ranker.run(language_model, layer_aggregator='bestlayer')

# print scores for each layer
print(result.layer_scores)
```

<details>
<summary><b>Layer Ranking</b>: Review the transferability scores for each layer.</summary>

The layer with index -41, which is the seventh from the bottom, gets the highest h-score.

```bash
INFO:transformer_ranker.ranker:microsoft/deberta-v2-xxlarge, score: 2.8912 (layer -41)
layer scores: {-1: 2.7377, -2: 2.8024, -3: 2.8312, -4: 2.8270, -5: 2.8293, -6: 2.7952, -7: 2.7894, -8: 2.7777, -9: 2.7490, -10: 2.7020, -11: 2.6537, -12: 2.7227, -13: 2.6930, -14: 2.7187, -15: 2.7494, -16: 2.7002, -17: 2.6834, -18: 2.6210, -19: 2.6126, -20: 2.6459, -21: 2.6693, -22: 2.6730, -23: 2.6475, -24: 2.7037, -25: 2.6768, -26: 2.6912, -27: 2.7300, -28: 2.7525, -29: 2.7691, -30: 2.7436, -31: 2.7702, -32: 2.7866, -33: 2.7737, -34: 2.7550, -35: 2.7269, -36: 2.7723, -37: 2.7586, -38: 2.7969, -39: 2.8551, -40: 2.8692, -41: 2.8912, -42: 2.8530, -43: 2.8646, -44: 2.8655, -45: 2.8210, -46: 2.7836, -47: 2.6945, -48: 2.5153}
```

</details>

<details>
<summary><b>Comparison to Linear Probe</b>: Review the results from training a linear probe.</summary>

| Layer index | LogME   | H-score | Dev F1 (Linear) | Test F1 (Linear) |
|-------------|---------|---------|-----------------|------------------|
| -1          | 0.7320  | 2.7421  | 0.9011          | 0.8674           |
| -2          | 0.7811  | 2.8125  | 0.9035          | 0.8765           |
| -3          | 0.7986  | 2.8460  | 0.9064          | 0.8812           |
| -4          | 0.7993  | 2.8404  | 0.9057          | 0.8786           |
| -5          | 0.7993  | 2.8359  | 0.9063          | 0.8805           |
| -6          | 0.7803  | 2.8073  | 0.9039          | 0.8754           |
| -7          | 0.7749  | 2.7982  | 0.9002          | 0.8739           |
| -8          | 0.7695  | 2.7890  | 0.9017          | 0.8681           |
| -9          | 0.7579  | 2.7614  | 0.8999          | 0.8687           |
| -10         | 0.7415  | 2.7106  | 0.8996          | 0.8688           |
| -11         | 0.7231  | 2.6661  | 0.8979          | 0.8687           |
| -12         | 0.7458  | 2.7311  | 0.8932          | 0.8636           |
| -13         | 0.7303  | 2.7003  | 0.8981          | 0.8634           |
| -14         | 0.7483  | 2.7262  | 0.9011          | 0.8737           |
| -15         | 0.7593  | 2.7564  | 0.9005          | 0.8703           |
| -16         | 0.7300  | 2.7000  | 0.8957          | 0.8688           |
| -17         | 0.7222  | 2.6849  | 0.8944          | 0.8624           |
| -18         | 0.6875  | 2.6224  | 0.8875          | 0.8554           |
| -19         | 0.6816  | 2.6145  | 0.8845          | 0.8582           |
| -20         | 0.6942  | 2.6462  | 0.8890          | 0.8599           |
| -21         | 0.7136  | 2.6780  | 0.8954          | 0.8633           |
| -22         | 0.7275  | 2.6795  | 0.9021          | 0.8721           |
| -23         | 0.7192  | 2.6491  | 0.9043          | 0.8689           |
| -24         | 0.7399  | 2.7007  | 0.9043          | 0.8711           |
| -25         | 0.7306  | 2.6727  | 0.9094          | 0.8754           |
| -26         | 0.7400  | 2.6895  | 0.9090          | 0.8831           |
| -27         | 0.7582  | 2.7315  | 0.9098          | 0.8736           |
| -28         | 0.7642  | 2.7539  | 0.9096          | 0.8777           |
| -29         | 0.7727  | 2.7726  | 0.9154          | 0.8853           |
| -30         | 0.7621  | 2.7496  | 0.9076          | 0.8724           |
| -31         | 0.7746  | 2.7747  | 0.9133          | 0.8844           |
| -32         | 0.7823  | 2.7910  | 0.9115          | 0.8877           |
| -33         | 0.7790  | 2.7797  | 0.9181          | 0.8850           |
| -34         | 0.7746  | 2.7605  | 0.9141          | 0.8832           |
| -35         | 0.7609  | 2.7295  | 0.9135          | 0.8883           |
| -36         | 0.7794  | 2.7719  | 0.9149          | 0.8866           |
| -37         | 0.7695  | 2.7587  | 0.9172          | 0.8875           |
| -38         | 0.7949  | 2.7967  | 0.9176          | 0.8869           |
| -39         | 0.8219  | 2.8569  | 0.9225          | 0.8930           |
| -40         | 0.8276  | 2.8710  | 0.9232          | 0.8960           |
| -41         | 0.8354  | 2.8877  | 0.9274          | 0.8972           |
| -42         | 0.8189  | 2.8541  | 0.9239          | 0.8892           |
| -43         | 0.8267  | 2.8650  | 0.9215          | 0.8887           |
| -44         | 0.8241  | 2.8685  | 0.9163          | 0.8887           |
| -45         | 0.8024  | 2.8297  | 0.9089          | 0.8713           |
| -46         | 0.7792  | 2.7903  | 0.9105          | 0.8656           |
| -47         | 0.7333  | 2.7008  | 0.9006          | 0.8556           |
| -48         | 0.6505  | 2.5113  | 0.8640          | 0.8086           |

</details>

__Running Time:__ Layer ranking took 1.5 minutes, with 52 seconds for embedding and 33 seconds for scoring. The dataset is embedded once, and each hidden state is scored independently (_n_ estimations for _n_ LM layers).
This was performed on a GPU-enabled (A100) Colab Notebook.

## Summary

This markdown explains how to use two parameters: `estimator` and `layer_aggregator` when running the ranker. 
The library also supports ranking layers of a single LM.
