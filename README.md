# demo

Display math:
```math
e^{i\pi} + 1 = 0
```
and inline math $`a^2 + b^2 = c^2`$.

# Tokenizer

Start by installing these two packages

```shell
pip install datasets
pip install transformers
```

I recommend to start with the `opus_books` [https://huggingface.co/datasets/opus_books] dataset from `huggingface` however you can try larger datasets if you want to built a more robust model. I am training the model using 2080ti with 11GB available of RAM in the GPU. 

### Downloading Data

Using the `datasets` package’s `load_dataset()` function you can download the dataset from directly from `huggingface`. The first argument is the name of the dataset and 2nd argument is the two difference language translation. Here I am downloading English and French language translation. The last argument is the available set we want either train, test or validation. Since we are using this for learning purpose only, we will just download the `train` file and split this into `train` and `validation` as needed.

```python
from datasets import load_dataset
hf_dataset=load_dataset('opus_books',f'en-fr',split='train')
```

If you just print the `hf_dataset`, it will show total number of rows and the feature names.

```
Dataset({
    features: ['id', 'translation'],
    num_rows: 127085
})
```

We can access this as `dict` by passing the feature name and index of any element.

```python
hf_dataset['translation'][0]
```

```
{'en': 'The Wanderer', 'fr': 'Le grand Meaulnes'}
```

### Dataset Iterator

As you see, in order to process the entire dataset, we need to have a function to loop through each translation in a lazy way. Let's use `yield` instead of `return` to create a generator function. Below the `get_one_sentence()` function taken in the entire dataset and the language to return the sentence to. We are using `yield` instead of `return` since `yield` produces a value but retains the function's state, allowing it to resume from where it left off, this way we don’t have to duplicate the dataset in memory again.

```python
def traverse_sentences(dataset, lang):
    for row in dataset:
        yield row['translation'][lang]
```

We can test the above function to make sure its working as expected.

```python
print(next(traverse_sentences(hf_dataset,'en')))
print(next(traverse_sentences(hf_dataset,'fr')))
```

```
The Wanderer
Le grand Meaulnes
```

### Tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
```

Create an instance of the `WordLevel` `Tokenizer` and set `[UNK]` for unknown tokens. 

```python
tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
```

We want to split the sentences by white space before applying the tokenizer. Hence set the `pre_tokenizer` to `Whitespace()`

```python
tokenizer.pre_tokenizer=Whitespace()
```

Now we need to define the trainer. We will use 4 special tokens `"[UNK]","[PAD]","[SOS]","[EOS]"` for Unknown tokens, Paddings, Start & End of Sentences. Also set `min_frequency` to `2 ` or `3` to accept words when their frequency of occurrence is at least equal to `min_frequency`.

```python
trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
```

In order to train the `tokenizer` , we need to pass all the sentences one by one. We can use the `traverse_sentences` function. 

Invoke `train_from_iterator()` function by passing the function `get_one_sentence` and the `trainer`. Then `save` the tokenizer to local path. This code is very mostly copied from the documentation of `tokenizer` library. You can always refer the official documentation for additional details.

```python
tokenizer.train_from_iterator(traverse_sentences(dataset,lang),trainer=trainer)
tokenizer.save(str(tokenizer_path))
```

##### Let's put everything together in a single function. We will load the tokenizer if its already created otherwise we will create one. 

```python
from pathlib import Path
def init_tokenizer(dataset, lang):
    tokenizer_path = Path(f"tokenizer_{lang}.json")
    tokenizer_loaded = False

    if tokenizer_path.exists():
        # try loading tokenizer if exists
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer_loaded = True
        except:
            pass
    if not tokenizer_loaded:
        # initiate tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            traverse_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer
```

Question is why we need the tokenizer instance. The `Tokenizer` includes many built-in functions which can be directly invoked. Here is a glimpse. The most important ones will be `token_to_id()` and `id_to_token()`.

<img src="tokenizer.assets/image-20240312003930490.png" alt="image-20240312003930490" style="zoom: 33%;" />

Lets, see it in practice. The example code is self-explanatory. 

```python
# Get the vocabulary size
print(tokenizer_src.get_vocab_size())
# Get the word for random id
print(tokenizer_src.id_to_token(1234))
# Get the id for "learning"
print(tokenizer_src.token_to_id("learning"))
# Convert sentence to array of token ids
print(tokenizer_src.encode("i love learning").ids)
# Convert array of token ids to sentence.
print(tokenizer_src.decode([5552, 194, 3125]))
```

```
15698
pay
3125
[5552, 194, 3125]
i love learning
```





