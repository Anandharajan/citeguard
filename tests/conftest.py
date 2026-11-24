import os

# Unit tests are offline and deterministic: use the lexical backend, never download a model.
os.environ.setdefault("CITEGUARD_NLI", "lexical")
