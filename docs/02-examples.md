# Tutorial 2: Learn by Example

This tutorial shows many usage examples of TransformerRanker. The idea is that you see the library in action 
for many NLP tasks and some special cases, and pick up functionality along the way. 

It probably makes sense to first complete tutorial 1 to learn all the basic concepts in TransformerRanker. 

## Example 1: Named Entity Recognition (NER) on Tweets

In this example, find the best-suited LMs for English NER on tweets. The full code is:

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load the WNUT-17 dataset of English tweets annotated with NER labels
dataset = load_dataset('leondz/wnut_17')

# Load a list of popular 'base' models
language_models = prepare_popular_models('base')

# Initialize the ranker, but also let it know what the text and label column is
ranker = TransformerRanker(dataset, dataset_downsample=0.7)

# ... and run it with language models
results = ranker.run(language_models, batch_size=64)

print(results_ner)
```

**Explanation**: This is essentially the same code as our introductory example. The only difference is that we 
downsample to 70% (instead of the default 20%). The reason for this is that WNUT-17 is already rather small so
we don't need to downsample too much.

If you run this code, you should get: 

```console 
Rank 1. microsoft/deberta-v3-base: 2.4273
Rank 2. sentence-transformers/all-mpnet-base-v2: 2.4021
Rank 3. roberta-base: 2.389
Rank 4. typeform/distilroberta-base-v2: 2.3104
Rank 5. microsoft/mdeberta-v3-base: 2.2357
Rank 6. Twitter/twhin-bert-base: 2.2294
Rank 7. FacebookAI/xlm-roberta-base: 2.1782
Rank 8. google/electra-base-discriminator: 2.1614
Rank 9. bert-base-cased: 1.953
Rank 10. german-nlp-group/electra-base-german-uncased: 1.7527
Rank 11. sentence-transformers/all-MiniLM-L12-v2: 1.6729
Rank 12. Lianglab/PharmBERT-cased: 1.4505
Rank 13. distilbert-base-cased: 1.3736
Rank 14. KISTI-AI/scideberta: 1.1893
Rank 15. google/electra-small-discriminator: 1.1288
Rank 16. SpanBERT/spanbert-base-cased: 1.0637
Rank 17. dmis-lab/biobert-base-cased-v1.2: 0.9517
```



## Example 2: Part-of-Speech Tagging 

In this example, we find the best-suited LMs for English part-of-speech tagging. The full code is:

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load the English part of the universal dependencies dataset
dataset_pos = load_dataset('universal-dependencies/universal_dependencies', 'en_lines')

# Load a list of popular 'base' models
language_models = prepare_popular_models('base')

# Initialize the Ranker, but also let it know what the text and label column is
ranker = TransformerRanker(dataset_pos,
                           dataset_downsample=0.5,
                           text_column="tokens", # in this dataset, the text column is labeled "tokens"
                           label_column="upos", # this dataset has many layers of annotation - we select "upos"
                           )

# ... and run it
results = ranker.run(language_models, batch_size=64)

print(results)
```

**Note:** You may need to `pip install connlu` to run this example.

**Explanation**: This example is a bit more complicated since we use the Universal Dependencies (UD) dataset.
UD contains many languages (English, German, Japanese, etc.) and annotates each sentence with many different layers of 
annotation (part-of-speech, universal part-of-speech, lemmas, morphology, dependency trees, etc.). 
For this reason, we pass `label_column="upos"` to the ranker, to let it know we want to predict 
universal part-of-speech tags (and not other annotations like lemmas).

If you run this code, you should get: 

```console
Rank 1. FacebookAI/xlm-roberta-base: 11.6609
Rank 2. microsoft/mdeberta-v3-base: 11.5872
Rank 3. Twitter/twhin-bert-base: 11.5575
Rank 4. google/electra-base-discriminator: 11.4565
Rank 5. microsoft/deberta-v3-base: 11.4192
Rank 6. typeform/distilroberta-base-v2: 11.4149
Rank 7. bert-base-cased: 11.3194
Rank 8. distilbert-base-cased: 11.1877
Rank 9. roberta-base: 11.1499
Rank 10. Lianglab/PharmBERT-cased: 11.1391
Rank 11. dmis-lab/biobert-base-cased-v1.2: 11.1363
Rank 12. SpanBERT/spanbert-base-cased: 11.0336
Rank 13. sentence-transformers/all-mpnet-base-v2: 10.9522
Rank 14. german-nlp-group/electra-base-german-uncased: 10.8265
Rank 15. sentence-transformers/all-MiniLM-L12-v2: 10.3615
Rank 16. google/electra-small-discriminator: 10.2858
Rank 17. KISTI-AI/scideberta: 10.1374
```

<details>
  <summary><b>Additional info</b>: How to select the "label_column"</summary>
  
Since there are so many layers of annotation, we need to tell TransformerRanker which defines the task we 
want to solve. First, we print the dataset to understand it better: 

```python
from datasets import load_dataset

# Load Universal Dependencies PoS dataset
dataset_pos = load_dataset('universal-dependencies/universal_dependencies', 'en_lines')

# Inspect the dataset
print(dataset_pos)
```

This outputs: 

```console 
DatasetDict({
    train: Dataset({
        features: ['idx', 'text', 'tokens', 'lemmas', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc'],
        num_rows: 3176
    })
    validation: Dataset({
        features: ['idx', 'text', 'tokens', 'lemmas', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc'],
        num_rows: 1032
    })
    test: Dataset({
        features: ['idx', 'text', 'tokens', 'lemmas', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc'],
        num_rows: 1035
    })
})
```

... which tells us that this dataset contains annotations such as "lemmas", 
"upos" (universal part-of-speech tags) and many others. 

</details>

## Example 3: Ranking Large Models

So far, we used our 'base' list of recommended language models. We also provide
a list of large models, and of course encourage users to try bigger models if 
you have the compute. 

Let's rank large models on the task of question classification: 

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load a dataset
dataset = load_dataset('trec')

# Load a predefined models or create your own list of models
language_models = prepare_popular_models('large')
print(language_models)

# Initialize the ranker with the dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# Run it with your models
results = ranker.run(language_models, batch_size=64)

print(results)
```

**Explanation**: Here, we select `prepare_popular_models('large')` for our suggested list of 11 bigger LMs to 
try. But you are free to extend this list by any other LM you'd like to try. We also pass 
`batch_size=64` because on this dataset with a T4 GPU, there is enough memory. If you run into 
memory issues, try passing `batch_size=16` or lower.

If you run this code, you should get: 

```console
Rank 1. microsoft/deberta-v3-large: 4.3306
Rank 2. google/electra-large-discriminator: 4.3293
Rank 3. deepset/gelectra-large: 4.2699
Rank 4. roberta-large: 4.2237
Rank 5. FacebookAI/xlm-roberta-large: 4.1913
Rank 6. Twitter/twhin-bert-large: 4.1595
Rank 7. bert-large-uncased: 4.1389
Rank 8. dmis-lab/biobert-large-cased-v1.1: 4.1111
Rank 9. sentence-transformers/all-mpnet-base-v2: 4.0739
Rank 10. microsoft/mdeberta-v3-base: 4.0091
Rank 11. KISTI-AI/scideberta: 3.7448
```


## Example 4: Text Pair Classificatiom

Some NLP do not classify a single text, but rather two. For instance, the task of 
recognizing textual entailment has two inputs: a premise and a hypothesis. 

You can use TransformerRanker also for text pair tasks, but you need to specify which
columns in the dataset belong to the text pair: 

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load 'rte' Recognizing Textual Entailment dataset
dataset = load_dataset('SetFit/rte')

# Use smaller models to test on cpu
language_models = prepare_popular_models('base')

# Initialize the ranker
ranker = TransformerRanker(dataset=dataset,
                           text_column="text1",
                           text_pair_column="text2",  # let it know where the second column is
                           label_column="label",
                           )

# ... and run it
result = ranker.run(models=language_models, batch_size=32)

print(result)
```

**Explanation:** In this case, we specify two text columns by passing both `text_column="text1"`
and `text_pair_column="text2"`. This is necessary because the textual entailment dataset
defines two textual inputs. 

If you run this code, you should get: 

```console
Rank 1. microsoft/deberta-v3-base: 0.4164
Rank 2. microsoft/mdeberta-v3-base: 0.3817
Rank 3. google/electra-base-discriminator: 0.3756
Rank 4. roberta-base: 0.3727
Rank 5. FacebookAI/xlm-roberta-base: 0.3358
Rank 6. german-nlp-group/electra-base-german-uncased: 0.3345
Rank 7. Twitter/twhin-bert-base: 0.3306
Rank 8. sentence-transformers/all-mpnet-base-v2: 0.3127
Rank 9. SpanBERT/spanbert-base-cased: 0.3043
Rank 10. typeform/distilroberta-base-v2: 0.3035
Rank 11. KISTI-AI/scideberta: 0.2966
Rank 12. bert-base-cased: 0.2953
Rank 13. dmis-lab/biobert-base-cased-v1.2: 0.291
Rank 14. Lianglab/PharmBERT-cased: 0.2783
Rank 15. distilbert-base-cased: 0.2781
Rank 16. sentence-transformers/all-MiniLM-L12-v2: 0.189
Rank 17. google/electra-small-discriminator: 0.1531
```


<details>
  <summary><b>Additional info</b>: How to select the two text columns</summary>
  
First, we print the RTE dataset to understand it better: 

```python
from datasets import load_dataset

# Load Universal Dependencies PoS dataset
dataset = load_dataset('SetFit/rte')

# Inspect the dataset
print(dataset)
```

This outputs: 

```console 
DatasetDict({
    train: Dataset({
        features: ['text1', 'text2', 'label', 'idx', 'label_text'],
        num_rows: 2490
    })
    validation: Dataset({
        features: ['text1', 'text2', 'label', 'idx', 'label_text'],
        num_rows: 277
    })
    test: Dataset({
        features: ['text1', 'text2', 'label', 'idx', 'label_text'],
        num_rows: 3000
    })
})
```

... which tells us that this dataset has two text column labeled "text1" and "text2" respectively.

</details>

