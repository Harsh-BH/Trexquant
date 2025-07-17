import string

# --- Vocabulary constants for both training and inference ---
LETTERS = list(string.ascii_lowercase)
LETTER2IDX = {c: i for i, c in enumerate(LETTERS)}
UNK_IDX = len(LETTER2IDX)             # index for “unknown” token
VOCAB_SIZE = UNK_IDX + 1             # 26 letters + 1 unk
EMBEDDING_DIM = 32
HIDDEN_DIM = 128