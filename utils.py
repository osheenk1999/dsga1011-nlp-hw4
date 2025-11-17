import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    """
    Custom transformation that makes the sentence more noisy in a realistic way:
    - Random synonym swaps (WordNet)
    - Occasional character swaps (typos)
    - Random small punctuation insertions
    """
    import random
    from nltk.corpus import wordnet
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    random.seed(42)
    detok = TreebankWordDetokenizer()

    # Slightly different neighbor map and typo logic
    keyboard_map = {
        "a": "s", "s": "d", "d": "f", "f": "g",
        "q": "w", "w": "e", "e": "r", "r": "t",
        "z": "x", "x": "c", "c": "v", "v": "b",
        "i": "o", "o": "p", "k": "l", "m": "n"
    }

    def synonym_swap(word):
        syns = wordnet.synsets(word)
        if not syns:
            return word
        lemmas = [l.name().replace("_", " ") for s in syns for l in s.lemmas()]
        # choose synonym different from original
        candidates = [s for s in lemmas if s.lower() != word.lower()]
        return random.choice(candidates) if candidates else word

    def inject_typo(word):
        if len(word) < 3:
            return word
        pos = random.randint(0, len(word) - 1)
        ch = word[pos].lower()
        if ch in keyboard_map:
            word = word[:pos] + keyboard_map[ch] + word[pos + 1:]
        return word

    tokens = word_tokenize(example["text"])
    transformed = []

    for w in tokens:
        p = random.random()
        # synonym replacement (~30%)
        if w.isalpha() and p < 0.3:
            w = synonym_swap(w)
        # typo injection (~20%)
        elif w.isalpha() and 0.3 <= p < 0.5:
            w = inject_typo(w)
        # punctuation insertion (~8%)
        elif p > 0.92:
            w = w + random.choice(["!", "?", "...", ","])
        transformed.append(w)

    example["text"] = detok.detokenize(transformed)
    return example
