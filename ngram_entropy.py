import math
import gc
import time
from collections import Counter
import pickle
import os

START_N = 1
END_N = 20
CORPUS_FILE = "corpus.txt"
ENTROPY_LOG_FILE = "entropy_results.txt"
MAX_CONTEXTS_IN_MEMORY = 1_000_000
BATCH_FOLDER = "ngram_batches"
LOG_BASE = 2


def clean_text(text):
    return ''.join(ch for ch in text.lower() if 'a' <= ch <= 'z')

def ensure_batch_folder():
    if not os.path.exists(BATCH_FOLDER):
        os.makedirs(BATCH_FOLDER)
    else:
        for f in os.listdir(BATCH_FOLDER):
            os.remove(os.path.join(BATCH_FOLDER, f))



def save_batches_streaming(corpus: bytes, n: int, max_contexts: int):
    """Stream through corpus once, saving batches when memory limit hit"""
    t = len(corpus)
    batch_num = 0
    batch_ngrams = Counter()
    batch_nplus1 = Counter()
    
    for i in range(t - n):
        context = corpus[i:i+n]
        next_letter = corpus[i+n:i+n+1]
        
        batch_ngrams[context] += 1
        batch_nplus1[(context, next_letter)] += 1
        
        if len(batch_ngrams) >= max_contexts:
            batch_file = os.path.join(BATCH_FOLDER, f"batch_{batch_num}.pkl")
            with open(batch_file, "wb") as f:
                pickle.dump((batch_ngrams, batch_nplus1), f)
            print(f"  Saved batch {batch_num} with {len(batch_ngrams):,} contexts")
            
            batch_num += 1
            batch_ngrams = Counter()
            batch_nplus1 = Counter()
            gc.collect()
    
    if batch_ngrams:
        batch_file = os.path.join(BATCH_FOLDER, f"batch_{batch_num}.pkl")
        with open(batch_file, "wb") as f:
            pickle.dump((batch_ngrams, batch_nplus1), f)
        print(f"  Saved final batch {batch_num} with {len(batch_ngrams):,} contexts")
        batch_num += 1
    
    return batch_num, t


def compute_entropy(num_batches, t, n, log_base=2):
    """
    Pass 1: Aggregate total counts per context across all batches
    Pass 2: Compute entropy using aggregated counts
    """
    log_fn = math.log2 if log_base==2 else lambda x: math.log(x, log_base)
    t_minus_n = t - n
    correction_numerator = t_minus_n + 1
    correction_denominator = t_minus_n
    
    # PASS 1: Build global count maps incrementally
    print("\nPass 1: Aggregating counts across batches...")
    global_ngrams = Counter()
    global_nplus1 = Counter()
    
    for batch_num in range(num_batches):
        batch_file = os.path.join(BATCH_FOLDER, f"batch_{batch_num}.pkl")
        with open(batch_file, "rb") as f:
            batch_ngrams, batch_nplus1 = pickle.load(f)
        
        global_ngrams.update(batch_ngrams)
        global_nplus1.update(batch_nplus1)
        
        del batch_ngrams, batch_nplus1
        gc.collect()
        
        print(f"  Processed batch {batch_num}/{num_batches-1}")
    
    # PASS 2: Compute entropy from aggregated counts
    print("\nPass 2: Computing entropy...")
    total_entropy = 0.0
    
    # Pre-build letter list ONCE
    letters = [bytes([i]) for i in range(ord('a'), ord('z')+1)]


    for idx, (context, n_context) in enumerate(global_ngrams.items()):
        if idx % 100000 == 0:
            print(f"  Processed {idx:,}/{len(global_ngrams):,} contexts...")
        context_contrib = 0.0
        for letter_byte in letters:
            key = (context, letter_byte)
            if key in global_nplus1:
                n_context_letter = global_nplus1[key]
                numerator = n_context_letter * correction_numerator
                denominator = n_context * correction_denominator
                context_contrib += n_context_letter * log_fn(numerator / denominator)
        total_entropy += context_contrib
    

    num_contexts = len(global_ngrams)
    entropy = -(1.0 / t_minus_n) * total_entropy
    
    return entropy, num_contexts


def main():
    ensure_batch_folder()

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = clean_text(f.read()).encode('ascii')

    results = []

    for n in range(START_N, END_N+1):
        print("\n" + "="*60)
        print(f"PROCESSING N = {n}")
        print("="*60)
        start_time = time.time()

        # Save batches
        num_batches, t = save_batches_streaming(corpus, n, MAX_CONTEXTS_IN_MEMORY)
        
        # Compute entropy
        entropy, num_contexts = compute_entropy(num_batches, t, n, LOG_BASE)

        duration = time.time() - start_time
        duration_str = f"{duration/60:.1f} min"

        print(f"\n✓ N={n} COMPLETE")
        print(f"Contexts: {num_contexts:,}, H(X|context)={entropy:.6f} bits, Time={duration_str}")

        results.append({
            'n': n,
            'contexts': num_contexts,
            'entropy': entropy,
            'duration': duration_str
        })

        # Cleanup batches
        for batch_num in range(num_batches):
            batch_file = os.path.join(BATCH_FOLDER, f"batch_{batch_num}.pkl")
            if os.path.exists(batch_file):
                os.remove(batch_file)
        gc.collect()

        # Save results
        with open(ENTROPY_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("N-gram Conditional Entropy Results\n")
            f.write("="*60 + "\n")
            for r in results:
                f.write(f"N={r['n']}, Contexts={r['contexts']:,}, H={r['entropy']:.6f}, Time={r['duration']}\n")

    print("\nALL DONE!")
    for r in results:
        print(f"N={r['n']}, Contexts={r['contexts']:,}, H={r['entropy']:.6f}, Time={r['duration']}")

if __name__ == "__main__":
    main()

