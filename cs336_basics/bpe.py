# cs336_basics/bpe.py
from collections import Counter
from typing import List, Tuple, Dict, Iterable
# from pretokenization_example import find_chunk_boundaries
import regex as re
import os

def pre_tokenization(
    chunks: Iterable[str]
) -> Counter[tuple[bytes, ...]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_counts = Counter()
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pre_token = match.group().encode("utf-8")
            if len(pre_token) != 1:
                # turn the bytestring object into a tuple of bytestring objects of single byte
                token_counts[tuple(bytes([x]) for x in pre_token)] += 1
    return token_counts

def save_vocab(vocab: dict[int, bytes], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for token_id in sorted(vocab):
            token_bytes = vocab[token_id]
            f.write(f"{token_id} {token_bytes.hex()}\n")

def save_merges(merges: list[tuple[bytes, bytes]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the corpus at input_path.
    Returns:
        vocab: dict[int, bytes] mapping token ID -> token bytes
        merges: list of byte-pair tuples representing BPE merges
    """

    with open(input_path, "rb") as f:
        corpus_bytes = f.read()
    
    text = corpus_bytes.decode("utf-8", errors="ignore")

    # 1. vocabulary initialization
    vocab_count = 256
    vocab = {i: bytes([i]) for i in range(vocab_count)}

    # 2. removing special tokens
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    chunks = re.split(pattern, text)

    vocab.update({vocab_count + i: token.encode("utf-8") for i, token in enumerate(special_tokens)})
    vocab_count += len(special_tokens)

    # 3. pre-tokenization
    token_counts = pre_tokenization(chunks) 

    # 4. compute BPE merges
    merges = []
    # 4.1. counts every pair
    pair_counts = Counter()
    for pre_token, count in token_counts.items():
        for left, right in zip(pre_token, pre_token[1:]):
            pair_counts[(left, right)] += count

    for index in range(vocab_size - vocab_count):
        # 4.1. identify the pair with the highest frequency
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
        merges.append(best_pair)
        new_merge_token = best_pair[0] + best_pair[1]
        # 4.2. merge the pair in `pre_token`
        for pre_token, count in list(token_counts.items()):
            new_tokens = []
            i = 0
            merge = False
            length = len(pre_token)
            while i < length:
                if i < length-1 and (pre_token[i], pre_token[i+1]) == best_pair:
                    new_tokens.append(pre_token[i] + pre_token[i+1])
                    merge = True
                    # 4.1. update `pair_count`
                    if i > 0:
                        pair_counts[(pre_token[i-1], new_merge_token)] += count
                        pair_counts[(pre_token[i-1], pre_token[i])] -= count
                    if i < length-2:
                        pair_counts[(new_merge_token, pre_token[i+2])] += count
                        pair_counts[(pre_token[i+1], pre_token[i+2])] -= count
                    i += 2
                else:
                    new_tokens.append(pre_token[i])
                    i += 1
            if merge:
                token_counts[tuple(new_tokens)] += token_counts[pre_token]
                del token_counts[pre_token]
        # 4.3. add the new merged token to the vocabulary
        vocab.update({vocab_count: new_merge_token})
        vocab_count += 1
        # 4.1. update `pair_count`
        del pair_counts[best_pair]

    return vocab, merges

def main():
    data_name = "test2"
    input_path = f"data/{data_name}.txt"        # path to your text file
    vocab_size = 1000                        # desired vocabulary size
    special_tokens = ["<|endoftext|>"]      # your special tokens

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    save_vocab(vocab, f"result/{data_name}/vocab.txt")
    save_merges(merges, f"result/{data_name}/merges.txt")

if __name__ == "__main__":
    main()