import os
import re

def gather_txt_files(folder):
    """Gather all .txt files inside a folder."""
    txt_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    return txt_files

def clean_text(text):
    """Keep only lowercase a-z characters."""
    return re.sub(r'[^a-z]', '', text.lower())

def build_corpus(data_folder, output_file="corpus.txt"):
    txt_files = gather_txt_files(data_folder)
    print(f"Found {len(txt_files)} text files.")

    with open(output_file, "w", encoding="utf-8") as corpus:
        for path in txt_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    cleaned = clean_text(content)
                    corpus.write(cleaned)
            except Exception as e:
                print(f"Error reading {path}: {e}")

    print(f"Corpus created at: {output_file}")

if __name__ == "__main__":
    build_corpus("data")
