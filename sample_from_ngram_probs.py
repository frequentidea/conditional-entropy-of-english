import string
import csv
from collections import Counter, defaultdict


N = 1  # number of previous letters to condition on (e.g., 1 = bigram, 2 = trigram, etc.)
CORPUS_FILE = "corpus.txt"
OUTPUT_FILE = f"ngram_{N}_conditional_probabilities.csv"


def clean_text(text):
    return ''.join(ch for ch in text.lower() if ch in string.ascii_lowercase)


def compute_ngram_probabilities(corpus, n):
    """Compute conditional probabilities of a letter given the previous n letters."""
    # Collect all n-gram and (n+1)-gram counts
    ngram_counts = Counter()
    nplus1_counts = Counter()
    t = len(corpus)

    for i in range(len(corpus) - n):
        context = corpus[i:i + n]          # previous n letters
        next_letter = corpus[i + n]        # next letter
        ngram_counts[context] += 1
        nplus1_counts[(context, next_letter)] += 1

    # Compute conditional probabilities
    probs = defaultdict(dict)
    for (context, next_letter), count in nplus1_counts.items():
        probs[context][next_letter] = (count / ngram_counts[context]) * (t / (t - 1))

    return probs


def write_probabilities_to_csv(probs, output_file):
    """Write probabilities to CSV."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Context"] + list(string.ascii_lowercase)
        writer.writerow(header)

        for context in sorted(probs.keys()):
            row = [context]
            for letter in string.ascii_lowercase:
                p = probs[context].get(letter, 0.0)
                row.append(f"{p:.6f}")
            writer.writerow(row)

    print(f"Saved conditional probabilities to {output_file}")


def main():
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        text = clean_text(f.read())

    print(f"Loaded corpus with {len(text)} characters.")
    print(f"Computing {N}-gram conditional probabilities...")

    probs = compute_ngram_probabilities(text, N)
    write_probabilities_to_csv(probs, OUTPUT_FILE)


if __name__ == "__main__":
    main()
