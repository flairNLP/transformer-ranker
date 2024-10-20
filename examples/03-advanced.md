# Transferability Estimation

_How is the suitability of language models estimated?_

Language models are evaluated by how well their embeddings match a specific downstream task, without adjusting the model's weights.
Embeddings from different layers are scored using a fast, lightweight method that doesn't need extra training.
The goal is to rank the models to predict their performance after fine-tuning.

### How it works

Estimating a language model's transferability involves two steps, repeated for each model:

1. **Extract embeddings**: The model acts as a fixed feature extractor. Its weights remain unchanged, and embeddings from different layers are collected during forward passes.
2. **Score embeddings**: The embeddings are scored with estimators to measure how well they fit the target task.

### Estimation Methods

The library provides three methods for estimating transferability:

- _k-Nearest Neighbors (k-NN)_: Uses distance metrics (e.g., Euclidean distance) to measure how closely embeddings from the same class are clustered. We mimic 1-fold cross-validation by calculating pairwise distances and excluding self-distances in the top _k_ search. [See k-NN code here](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/nearestneighbors.py).
- _H-score_: Measures the feature-wise variance between embeddings of different classes. High variance with low feature redundancy suggests strong transferability. [See H-score code here](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/hscore.py).
- _LogME_: Computes the marginal likelihood of a linear model fitted to the embeddings without actual training. It optimizes two parameters (alpha and beta) to provide a score reflecting the alignment of embeddings with the task. [See LogME code here](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/logme.py).

We rely on two state-of-the-art metrics: LogME and H-score (with shrinkage-based improvements). More metrics are out there, so there might be room for improvement!

### Applicability of Metrics

These metrics are mainly for classification tasks.
For text pair tasks like entailment, the problem is handled as classification by combining two sentences with a separator token.
For regression tasks, only LogME is recommended.

| **Transferability Metric** | **Classification** | **Regression** | **Generation** |
|----------------------------|--------------------|----------------|----------------|
| k-NN                       | ✓                  | ✓              | ✗              |
| H-score                    | ✓                  | ✗              | ✗              |
| LogME                      | ✓                  | ✓              | ✗              |

To use a different transferability metric, set the `estimator` parameter when running the ranker:

```python3
result = ranker.run(language_models, estimator="logme")
```

### Layer selection

When using language models as feature extractors, the question is which layer's features to use for better rankings. The library offers three options:

- _lastlayer_: Uses the embeddings from the final hidden layer. Models are ranked by how well the final layer's embeddings are suited to the task.
- _bestlayer_: Scores all layers and selects the one with the highest. Models are ranked by their best-performing layers for a task.
- _layermean_: Averages all hidden layers into a single embedding. Models are ranked based on all-layer average embedding's score.

To use a different layer selection method, add the `layer_aggregation` parameter in the run method:

```python3
result = ranker.run(language_models, estimator="logme", layer_aggregator="bestlayer")
```

# Layerwise Analysis

The `bestlayer` option is helpful even for a single model, as it shows transferability scores for each layer.
This allows you to quickly see how well each layer works for your dataset, indicating which layers best separate the classes.

### Example: deberta-v2-xxlarge X CoNLL (NER)

For example, you can use the DeBERTa-xxlarge model and run the best layer search on the CoNLL (NER) dataset.
The goal is to identify which layer's embeddings best separate four entity classes.
You can do this by loading the model and setting `layer_aggregator='bestlayer'` when running the ranker.

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
result = ranker.run(models=language_model, layer_aggregator='bestlayer')
```

### Results

Each layer's score is logged, and the highest one shows the best layer for the task.
Layer -41 (7th from the bottom) provides the best separation for the four entities.

```bash
INFO:transformer_ranker.ranker:microsoft/deberta-v2-xxlarge, score: 2.8912 (layer -41)
layerwise scores: {-1: 2.7377, -2: 2.8024, -3: 2.8312, -4: 2.8270, -5: 2.8293, -6: 2.7952, -7: 2.7894, -8: 2.7777, -9: 2.7490, -10: 2.7020, -11: 2.6537, -12: 2.7227, -13: 2.6930, -14: 2.7187, -15: 2.7494, -16: 2.7002, -17: 2.6834, -18: 2.6210, -19: 2.6126, -20: 2.6459, -21: 2.6693, -22: 2.6730, -23: 2.6475, -24: 2.7037, -25: 2.6768, -26: 2.6912, -27: 2.7300, -28: 2.7525, -29: 2.7691, -30: 2.7436, -31: 2.7702, -32: 2.7866, -33: 2.7737, -34: 2.7550, -35: 2.7269, -36: 2.7723, -37: 2.7586, -38: 2.7969, -39: 2.8551, -40: 2.8692, -41: 2.8912, -42: 2.8530, -43: 2.8646, -44: 2.8655, -45: 2.8210, -46: 2.7836, -47: 2.6945, -48: 2.5153}
```

### Efficiency

Layer ranking is computationally efficient.
The dataset is embedded once, and each hidden state is scored independently (_n_ estimations for _n_ transformer layers).
Since the estimators are lightweight, scoring models with many layers is fast.

For DeBERTa-xxlarge (48 layers), the whole process took 1.5 minutes: 52 seconds for embedding and 33 seconds for scoring.
This was done on a GPU-enabled (A100) Colab Notebook.

### Why not just train a linear probe?

Training a linear layer can be an alternative to estimators, but it has two downsides:

- Slower runtime: It's slower than using transferability metrics, especially for large models with many layers (like 48 layers)
- Hyperparameter search: Training requires tuning parameters like learning rate, batch size, and number of epochs, while metrics like H-score don't have any hyperparameters.

<details> <summary> Training a linear probe on DeBERTa-xxlarge layers using CoNLL <br> </summary>


| Layer index | CoNLL dev | CoNLL test |
|-------------|-----------|------------|
| -1          | 0.9011    | 0.8674     |
| -2          | 0.9035    | 0.8765     |
| -3          | 0.9064    | 0.8812     |
| -4          | 0.9057    | 0.8786     |
| -5          | 0.9063    | 0.8805     |
| -6          | 0.9039    | 0.8754     |
| -7          | 0.9002    | 0.8739     |
| -8          | 0.9017    | 0.8681     |
| -9          | 0.8999    | 0.8687     |
| -10         | 0.8996    | 0.8688     |
| -11         | 0.8979    | 0.8687     |
| -12         | 0.8932    | 0.8636     |
| -13         | 0.8981    | 0.8634     |
| -14         | 0.9011    | 0.8737     |
| -15         | 0.9005    | 0.8703     |
| -16         | 0.8957    | 0.8688     |
| -17         | 0.8944    | 0.8624     |
| -18         | 0.8875    | 0.8554     |
| -19         | 0.8845    | 0.8582     |
| -20         | 0.8890    | 0.8599     |
| -21         | 0.8954    | 0.8633     |
| -22         | 0.9021    | 0.8721     |
| -23         | 0.9043    | 0.8689     |
| -24         | 0.9043    | 0.8711     |
| -25         | 0.9094    | 0.8754     |
| -26         | 0.9090    | 0.8831     |
| -27         | 0.9098    | 0.8736     |
| -28         | 0.9096    | 0.8777     |
| -29         | 0.9154    | 0.8853     |
| -30         | 0.9076    | 0.8724     |
| -31         | 0.9133    | 0.8844     |
| -32         | 0.9115    | 0.8877     |
| -33         | 0.9181    | 0.8850     |
| -34         | 0.9141    | 0.8832     |
| -35         | 0.9135    | 0.8883     |
| -36         | 0.9149    | 0.8866     |
| -37         | 0.9172    | 0.8875     |
| -38         | 0.9176    | 0.8869     |
| -39         | 0.9225    | 0.8930     |
| -40         | 0.9232    | 0.8960     |
| -41         | 0.9274    | 0.8972     |
| -42         | 0.9239    | 0.8892     |
| -43         | 0.9215    | 0.8887     |
| -44         | 0.9163    | 0.8887     |
| -45         | 0.9089    | 0.8713     |
| -46         | 0.9105    | 0.8656     |
| -47         | 0.9006    | 0.8556     |
| -48         | 0.8640    | 0.8086     |

We use the F1-micro average to report scores for both development and test splits.

The layerwise scores of linear probing strongly correlate with H-scores.

</details>

## Summary

We explained how to estimate the transferability of language models. 
The process involves embedding the dataset once through the model, then scoring the embeddings using transferability metrics.
You can choose the metric by setting the `estimator` parameter.

The library offers three options for pooling embeddings from different layers using `layer_aggregator` parameter.
The `layer_aggregator='bestlayer'` option is useful even when inspecting a single model.
By quickly scoring each layer, it offers valuable insights into how well different layers separate embeddings for your classification tasks.
