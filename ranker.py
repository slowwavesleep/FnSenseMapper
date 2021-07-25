import json
import re
from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd
import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class CandidateRanker:

    def __init__(self,
                 data_path: str,
                 out_path: str,
                 no_header: bool,
                 model_name: str,
                 n_candidates: int,
                 substring_condition: bool = True,
                 cutoff_similarity: Optional[float] = None,
                 ):

        self.data_path = data_path
        self.out_path = out_path
        self.model_name = model_name
        self.no_header = no_header

        self.data = None
        self.lu_ids = None
        self.init_data()

        self.n_candidates = n_candidates
        self.substring_condition = substring_condition
        self.cutoff_similarity = cutoff_similarity

        self.sentence_embedder = None
        self.init_embedder()

    @staticmethod
    def clean_entry(entry_name: str) -> str:
        wiki_sub = re.sub(r"^.+:.+:", "", entry_name)
        wn_sub = re.sub(r"\#.\#\d$", "", wiki_sub)
        return wn_sub.replace("_", " ")

    @staticmethod
    def argsort_with_cutoff(x: np.ndarray, threshold: Union[int, float]) -> np.ndarray:
        idx, = np.where(x > threshold)
        return idx[np.argsort(x[idx])][::-1]

    def init_data(self):
        if self.no_header:
            self.data = pd.read_csv(self.data_path, header=None)
            self.data.columns = ["idLu",
                                 "word",
                                 "pos",
                                 "fnDefinition",
                                 "entryId",
                                 "entryName",
                                 "entrySource",
                                 "bnDefinition"]
        else:
            self.data = pd.read_csv(self.data_path)

        self.data.bnDefinition = self.data.bnDefinition.fillna("")
        self.data.fnDefinition = self.data.fnDefinition.fillna("")
        self.lu_ids = self.data.idLu.unique()

    def init_embedder(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_embedder = SentenceTransformer(self.model_name, device=device)

    def subset_generator(self):
        for lu_id in self.lu_ids:
            yield self.data.loc[self.data.idLu == lu_id]

    def process_subset(self, df_subset: pd.DataFrame) -> Dict[str, Union[int, str, List[float], List[str]]]:
        id_lu: int = df_subset.iloc[0].idLu
        fn_word: str = df_subset.iloc[0].word
        fn_pos: str = df_subset.iloc[0].pos
        fn_definition: str = df_subset.iloc[0].fnDefinition
        bn_definitions: List[str] = df_subset.bnDefinition.to_numpy()
        bn_ids: List[str] = df_subset.entryId.to_numpy()
        bn_names: List[str] = df_subset.entryName.to_numpy()

        n_candidates: int = max(min(len(bn_definitions), self.n_candidates), 1)

        encoded_fn_definition: np.ndarray = np.expand_dims(self.sentence_embedder.encode(fn_definition), axis=0)
        encoded_bn_definitions: np.ndarray = self.sentence_embedder.encode(bn_definitions)
        similarities: np.ndarray = cosine_similarity(encoded_fn_definition, encoded_bn_definitions).squeeze(0)

        # remove entries below threshold if minimal similarity is specified
        if self.cutoff_similarity is not None and 0 < self.cutoff_similarity < 1:
            candidate_indices: np.ndarray = self.argsort_with_cutoff(x=similarities,
                                                                     threshold=self.cutoff_similarity)[:n_candidates]
        else:
            candidate_indices: np.ndarray = np.argsort(similarities)[::-1][:n_candidates]

        scores: np.ndarray = similarities[candidate_indices]
        mapped_definitions: np.ndarray = bn_definitions[candidate_indices]
        mapped_ids: np.ndarray = bn_ids[candidate_indices]
        mapped_names: np.ndarray = bn_names[candidate_indices]

        if mapped_names.size != 0 and self.substring_condition:
            substring_indices = [True if fuzz.partial_ratio(fn_word.lower(), self.clean_entry(name.lower())) > 90
                                 else False
                                 for name in mapped_names]
            scores = scores[substring_indices]
            mapped_definitions = mapped_definitions[substring_indices]
            mapped_ids = mapped_ids[substring_indices]
            mapped_names = mapped_names[substring_indices]

        return {"bn_ids": list(mapped_ids),
                "bn_names": list(mapped_names),
                "bn_definitions": list(mapped_definitions),
                "scores": [float(score) for score in scores],
                "id_lu": int(id_lu),
                "fn_word": fn_word,
                "fn_definition": fn_definition,
                "fn_pos": fn_pos}

    def write_candidates(self):
        with open(self.out_path, "w") as file:
            for subset in tqdm(self.subset_generator(), total=len(self.lu_ids)):
                file.write(json.dumps(self.process_subset(subset), ensure_ascii=False) + "\n")
