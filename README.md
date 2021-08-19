# FnSenseMapper
A tool to map FrameNet Lexical Units to BabelNet synsets using the distance between sentence embeddings between corresponding definitions


## Install and run
Create a virtual environment and install dependencies:
```
conda env -n mapper python=3.7
```
```
conda activate mapper
```
```
cd FnSenseMapper
```
```
pip install -r requirements.txt
```

Run the mapper:
```
python run.py
```

Note that the existing `bn_edges.csv` in `resources` is truncated due to Github's file size restrictions.

The following mapper parameters may be modified in `config.yml`:
- Maximum number of candidates to keep
- Cosine similarity cutoff threshold
- Whether to discard candidates where LU lemma is not a (partial )substring of BN entry name
