# N-gram Conditional Entropy

## Overview

This repository contains a set of Python scripts for analyzing the statistical structure of written English using n-gram models. The project focuses on computing conditional entropy for increasing context lengths.

To handle large context sizes efficiently, the implementation uses streaming, batching, and disk storage to avoid exceeding memory limits. The resulting conditional probabilities can also be used to generate text samples that mimic the english language (Kind of like a mini-LLM).



## Programming Language(s)

- **Python 3**
- Standard library modules (`collections`, `math`, `pickle`, `gc`, `os`, `csv`, etc.)


## Project Structure

- `build_corpus.py`  
  Gathers and cleans raw `.txt` files into a single character corpus.

- `ngram_entropy.py`  
  Computes conditional entropy for n-grams using a memory efficient batching approach.

- `sample_from_ngram_probs.py`  
  Samples text from previously computed conditional probability tables.



## Purpose of the Code

The main goals of this project are to:

- Quantify how predictable English text becomes as the length of known prior context increases.
- Compute conditional entropy for n-grams up to large values of *n*.
- Solving memory constraints when working with exponentially growing data.
- Generate new text sequences that statistically resemble the training corpus.



## Scale of the Project

- **Number of scripts:** 3
- **Functions:** ~15
- **Lines of code:** ~400



## Use of Object-Oriented Concepts

This project is primarily procedural rather than object-oriented, as the focus is on numerical analysis. Although, Encapsulation is used through functions and modular scripts.



## Use of Data Structures

- Dictionaries
- Tuples
- Lists
- Strings and byte arrays 
- Pickle files


## Interesting Algorithms and Design Choices

### Conditional Entropy Computation
- Computes conditional entropy using frequencies derived from the corpus.

### Streaming and Batching
- Each batch is saved to disk when a memory threshold is reached.


## My findings through this investigation

- Conditional entropy **decreases as context length increases**, reflecting increased predictability in English.
- Entropy eventually plateaus, indicating inherent randomness.
- However, due to the limited corpus size increasing N will eventually cause the entropy to reach 0
