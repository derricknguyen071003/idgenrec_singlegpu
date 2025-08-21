from typing import Dict, List

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
import sys
sys.setrecursionlimit(3000)  # or even higher if needed


class Trie(object):
    """
    A prefix tree (trie) to store valid token sequences (e.g., textual item IDs).
    This enables constrained decoding: the model can only generate sequences
    that follow the valid prefixes stored in this trie.
    """
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}  # Root of the trie structure
        self.len = 0  # Count of sequences stored
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)  # Recursively insert sequence
                self.len += 1

        # Optional: allow one trie to extend another (e.g., prefix extension or merging)
        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        """
        Append another trie to this trie.
        Useful for combining multiple valid prefix sources.
        """
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        """
        Add a single sequence to the trie.
        """
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        """
        Given a prefix, return the set of valid next tokens.
        This function is used during generation to restrict the vocabulary.
        """
        return Trie._get_from_trie(prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id)

    @staticmethod
    def load_from_dict(trie_dict):
        """
        Reconstruct a Trie from a saved dictionary structure.
        """
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        """
        Recursively insert a token sequence into the trie.
        """
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(prefix_sequence: List[int], trie_dict: Dict, append_trie=None, bos_token_id: int = None):
        """
        Recursively navigate the trie given a prefix.
        Return valid next tokens at the current depth.
        """
        if len(prefix_sequence) == 0:
            # We've reached the current prefix; return possible next tokens
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                # Merge with another trie (minus the BOS token)
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            # Fallback: use appended trie (e.g., backup or general prefix set)
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        """
        Iterator to yield all sequences stored in the trie.
        """
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(prefix_sequence + [next_token], trie_dict[next_token])
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)



def prefix_allowed_tokens_fn(candidate_trie):
    """
    Wrapper to create a token filtering function for beam search generation.
    HuggingFace's `generate()` will call this function at each step.
    """
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()  # Convert tensor to list of token IDs
        trie_out = candidate_trie.get(sentence)  # Get valid next tokens from trie
        return trie_out

    return prefix_allowed_tokens


def exact_match(predictions, targets, k):
    """
    Compute top-k exact match accuracy.
    predictions: flat list of k * batch_size predictions
    targets: ground-truth targets (1 per example)
    k: number of predictions per example
    """
    batched_predictions = []
    batch_length = len(targets)
    for b in range(batch_length):
        batched_predictions.append(predictions[b * k : (b + 1) * k])
    correct = 0
    for p, t in zip(batched_predictions, targets):
        if t in p:
            correct += 1

    return correct



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Create candidate ID sequences using tokenized natural-language item IDs
    candidates = [
        "3560", "5540", "1825", "1062", "6880", "1683", "3632", "9273",
        "2345", "1398", "2000", "5992", "3754", "3637", "3272", "1531",
    ]
    
    # Encode each ID as a token sequence like: [0, ..., ..., ..., 1]
    candidate_trie = Trie([[0] + tokenizer.encode(f"Beauty item {e}") for e in candidates])
    print(candidate_trie.trie_dict)

    # Input prompt: user purchase history, used to predict next item
    input_s = [
        "Here is the purchase history of Beauty user 1: Beauty item 1,2,3,4,5. I wonder what is the next recommended item for the user.",
        "Here is the purchase history of Beauty user 3: Beauty item 27,52,za97,2. I wonder what is the next recommended item for the user.",
    ]
    input_ids = tokenizer.batch_encode_plus(input_s, padding="longest", return_tensors="pt")["input_ids"]
    print(input_ids.size())
    # Constrain generation using our trie
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    output_ids = model.generate(
        input_ids,
        max_length=150,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=20,
        num_return_sequences=2,
    )

    print(output_ids.size())
    print(tokenizer.batch_decode(output_ids))

