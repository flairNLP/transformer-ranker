# Transferability Estimation

_How is the suitability of language models estimated?_

The suitability of language models is determined by evaluating the alignment of their embeddings with a specific downstream task.
This evaluation can be done in advance, without changing the model’s weights.
Embeddings are extracted from different layers of the model and scored using a lightweight method that is both fast and does not require additional training.
The goal is to rank the models in an order that closely matches their true performance after fine-tuning.

### How it works

Estimating language model's transferability has two main steps. These steps are repeated for each language model. 

1. **Extract embeddings**: The model is treated as a fixed feature extractor. Its weights remain unchanged, and we collect embeddings from various layers through forward passes.
2. **Score embeddings**: The embeddings are evaluated using estimators to assess how well they align with the target task.

### Estimation Methods

The library provides three methods for estimating transferability:

- _k-Nearest Neighbors (k-NN)_: Uses distance metrics (e.g., Euclidean distance) to measure how closely embeddings from the same class are clustered. We mimic 1-fold cross-validation by calculating pairwise distances and excluding self-distances in the top _k_ search. [See k-NN code here](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/nearestneighbors.py).
- _H-score_: Measures the feature-wise variance between embeddings of different classes. High variance with low redundancy suggests strong transferability. [See H-score code here](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/hscore.py).
- _LogME_: Calculates the marginal likelihood of a linear model fitted to the embeddings without actual training. It optimizes two parameters (alpha and beta) to provide a score reflecting the alignment of embeddings with the task. [See LogME code here](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/logme.py).

We use two trusted metrics, LogME and H-score (shrinkage-based version). More metrics are out there, so there might be room for improvement!

### Applicability of Metrics

These metrics are primarily for classification tasks.
For text pair tasks like entailment, the problem is treated as classification by concatenating two sentences with a separator token.
For regression tasks, only LogME is recommended.

| **Transferability Metric** | **Classification** | **Regression** | **Generation** |
|----------------------------|--------------------|----------------|----------------|
| k-NN                       | ✓                  | ✓              | ✗              |
| H-score                    | ✓                  | ✗              | ✗              |
| LogME                      | ✓                  | ✓              | ✗              |

To choose a different transferability metric, use the `estimator` parameter when running the ranker:

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

The _bestlayer_ option is useful even for a single language model, as it shows transferability scores for each layer.
It can be helpful to quickly inspect how well is each layer suited for your dataset.
To add, this indicates which layers separate the classes in your dataset better. 

### Example: deberta-v2-xxlarge X CoNLL (NER)

As an example, let's use the DeBERTa-xxlarge model and run the best layer search on the CoNLL (NER) dataset.
The goal is to identify which layer's embeddings best separate four entity classes.
This can be done by loading a single language model and setting the `layer_aggregator='bestlayer'` parameter when running the ranker.

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

Each layer’s score will be logged, and the highest-ranking layer will indicate the best layer for the task.
The layer -41 (7th from the bottom) offers the best separation for the four entities:

```bash
INFO:transformer_ranker.ranker:microsoft/deberta-v2-xxlarge, score: 2.8912 (layer -41)
layerwise scores: {-1: 2.7377, -2: 2.8024, -3: 2.8312, -4: 2.8270, -5: 2.8293, -6: 2.7952, -7: 2.7894, -8: 2.7777, -9: 2.7490, -10: 2.7020, -11: 2.6537, -12: 2.7227, -13: 2.6930, -14: 2.7187, -15: 2.7494, -16: 2.7002, -17: 2.6834, -18: 2.6210, -19: 2.6126, -20: 2.6459, -21: 2.6693, -22: 2.6730, -23: 2.6475, -24: 2.7037, -25: 2.6768, -26: 2.6912, -27: 2.7300, -28: 2.7525, -29: 2.7691, -30: 2.7436, -31: 2.7702, -32: 2.7866, -33: 2.7737, -34: 2.7550, -35: 2.7269, -36: 2.7723, -37: 2.7586, -38: 2.7969, -39: 2.8551, -40: 2.8692, -41: 2.8912, -42: 2.8530, -43: 2.8646, -44: 2.8655, -45: 2.8210, -46: 2.7836, -47: 2.6945, -48: 2.5153}
```

### Efficiency

Layer ranking is computationally efficient. 
The dataset is embedded once, and each hidden state is scored independently (_n_ estimations for _n_ transformer layers). 
Since the estimators are lightweight, scoring models with many layers is quick.

In the DeBERTa-xxlarge example, evaluating 48 layers took 1.5 minutes, including embedding and scoring (_embedding time_: 52 seconds, _scoring 48 layers_: 33 seconds).
The process involves extracting embeddings in a single pass and scoring each layer independently.
We used a GPU-enabled (A100) Colab Notebook.

### Why not just train a linear probe?

Training a linear layer can be an alternative to using estimators, but it has two shortcomings:

- Slower runtime: Training a linear layer is slower than calculating transferability metrics, especially for large models with many layers (e.g., 48 layers).
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

Here we explained how the transferability of language models is estimated.
It requires to embed the dataset through a language model once and then score the embeddings using transferability metrics. 
Choosing the estimator can be done by changing the 'estimator' parameter.
These metrics are applicable for classification and regression tasks. 

To answer which layers' embeddings result in best rankings, we offer three methods for layer selection.
The 'bestlayer' option is useful even when inspecting a single language model.
Given efficient implementation, we can quickly score each layer of a language model.
This gives insights of how different layers separate the embeddings for your classification tasks.
