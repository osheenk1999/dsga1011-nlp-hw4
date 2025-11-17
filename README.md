# DSGA 1011 — NLP Homework 4

This repository contains the full code for **Homework 4** of the NYU course *DS-GA 1011: Natural Language Processing*.  
The assignment covers two main parts:

1. **Text Classification with Data Transformations** (Part 1)  
2. **Text-to-SQL Generation with T5 Fine-Tuning** (Part 2)

---

## 📁 Repository Structure

| File | Description |
|------|--------------|
| `main.py` | Main training and evaluation script for **Part 1**, which uses a BERT-based classifier on the IMDB sentiment dataset. Includes code for training, evaluating, and applying transformations or data augmentation. |
| `utils.py` | Helper functions for **Part 1**, including data transformations such as synonym replacement, typos, and token-level noise for robustness testing and augmentation. |
| `load_data.py` | Data processing module for **Part 2**, which loads and tokenizes natural language–SQL pairs for T5 model training. |
| `train_t5.py` | Core training and evaluation script for **Part 2**, implementing T5 fine-tuning for text-to-SQL generation using the provided dataset and evaluating Record F1 and Exact Match (EM). |

---

## 🧩 Part 1: IMDB Classification with Data Augmentation

**Goal:**  
Train a binary sentiment classifier using a pre-trained `bert-base-cased` model on the IMDB dataset.  
Test robustness using custom text transformations and evaluate how augmentation affects performance on both the original and transformed test sets.

**Main features:**
- Implements `custom_transform()` for data corruption (typos, synonym swaps, and punctuation noise).
- Adds data augmentation via 5,000 randomly transformed training examples.
- Evaluates performance on both original and transformed data.

**Example commands:**
```bash
# Train baseline model
python3 main.py --train

# Evaluate baseline model
python3 main.py --eval --model_dir ./out

# Train with data augmentation
python3 main.py --train_augmented

# Evaluate augmented model on transformed test set
python3 main.py --eval_transformed --model_dir ./out_augmented
