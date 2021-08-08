# FnSenseMapper
A tool to map FrameNet Lexical Units to BabelNet synsets using the distance between sentence embeddings between corresponding definitions

Assuming that `bn_entries.csv` and `bn_edges.csv` are in `resources` the mapper can be run using the following command:

```
python run.py
```

The following mapper parameters may be modified in `config.yml`:
- Maximum number of candidates to keep
- Cosine similarity cutoff threshold
- Whether to discard candidates where LU lemma is not a substring of BN entry name
