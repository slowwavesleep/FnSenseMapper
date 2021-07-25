import json
from collections import defaultdict
from typing import List

import pandas as pd
from tqdm import tqdm

from constants import DATA_COLUMNS, RELATION_ENTRIES_FIELD_NAMES, EDGES_FIELD_NAMES, CANDIDATE_FIELD_NAMES, \
    ENTRY_ID_NAME, RELATION_NAME


class RelationGenerator:

    RELATIONS_TO_DROP = [
        "gloss_related_form_(disambiguated)",
        "gloss_related_form_(monosemous)",
        "semantically_related_form",
        "domain_of_synset_-_topic",
        "derivationally_related_form",
        "member_of_this_domain_-_topic",
        "member_of_this_domain_-_usage",
        "domain_of_synset_-_usage"
    ]

    def __init__(self,
                 edges_path: str,
                 candidates_path: str,
                 entries_path: str,
                 out_path: str,
                 no_header: bool):

        self.out_path = out_path

        self.entries_path = entries_path
        self.entries = None
        self.no_header = no_header

        self.edges_path = edges_path
        self.edges = None
        self.init_edges()

        self.candidates_path = candidates_path
        self.candidates = []
        self.init_candidates()

        self.lu2bn = defaultdict(lambda: [])
        self.init_lu2bn()

        self.lu_edges = None

        self.lu_relations = None

    def init_entries(self):
        if self.no_header:
            self.entries = pd.read_csv(self.entries_path, header=None)
            self.entries.columns = DATA_COLUMNS
        else:
            self.entries = pd.read_csv(self.entries_path)

        self.entries.bnDefinition = self.entries.bnDefinition.fillna("")
        self.entries.fnDefinition = self.entries.fnDefinition.fillna("")
        self.entries = self.entries[RELATION_ENTRIES_FIELD_NAMES]

    def init_edges(self):
        if self.no_header:
            self.edges = pd.read_csv(self.edges_path, header=None)
            self.edges.columns = EDGES_FIELD_NAMES
        else:
            self.edges = pd.read_csv(self.edges_path)
        self.edges = self.edges.loc[~self.edges.edges_string.isna()]
        self.edges["edges"] = self.edges.edges_string.apply(self.convert_edges)

    def init_candidates(self):
        with open(self.candidates_path) as file:
            for line in file:
                candidate = json.loads(line)
                if candidate[CANDIDATE_FIELD_NAMES["bn_names"]]:
                    self.candidates.append(candidate)

    def init_lu2bn(self):
        for candidate in self.candidates:
            id_lu = candidate[CANDIDATE_FIELD_NAMES["id_lu"]]
            bn_ids = candidate[CANDIDATE_FIELD_NAMES["bn_ids"]]
            for bn_id in bn_ids:
                self.lu2bn[bn_id].append(id_lu)

    def map_edges(self, edges_list):
        fn_relation_candidates = []
        for rel, bn_id in edges_list:
            id_lus = self.lu2bn[bn_id]
            if id_lus:
                for id_lu in id_lus:
                    fn_relation_candidates.append((rel, id_lu))
        return fn_relation_candidates

    def init_lu_edges(self):
        self.lu_edges = self.edges.copy()
        self.lu_edges["fn_candidates"] = self.lu_edges.entryId.apply(lambda x: self.lu2bn[x])

        self.lu_edges = self.lu_edges.fn_candidates.apply(pd.Series). \
            merge(self.lu_edges, left_index=True, right_index=True). \
            drop(["fn_candidates", "edges_string", "edges"], axis=1). \
            melt(id_vars=[ENTRY_ID_NAME["entryId"]], value_name=CANDIDATE_FIELD_NAMES["id_lu"]).drop(["variable"], axis=1)

        self.lu_edges = self.lu_edges.loc[~self.lu_edges.id_lu.isna()]
        self.lu_edges[CANDIDATE_FIELD_NAMES["id_lu"]] = self.lu_edges[CANDIDATE_FIELD_NAMES["id_lu"]].astype(int)

    def get_relations(self):
        if self.lu_edges is None:
            self.init_lu_edges()

        self.lu_edges = self.lu_edges.merge(self.edges, on=ENTRY_ID_NAME["entryId"])[[ENTRY_ID_NAME["entryId"], CANDIDATE_FIELD_NAMES["id_lu"], "edges"]]
        self.lu_edges.edges = self.lu_edges.edges.apply(self.map_edges)

    def generate_output(self):
        relations = []
        for row in tqdm(self.lu_edges.iterrows()):
            this_id_lu = row[1][CANDIDATE_FIELD_NAMES["id_lu"]]
            edges = row[1]["edges"]
            if edges:
                for relation, other_id_lu in edges:
                    relations.append({
                        f"{CANDIDATE_FIELD_NAMES['id_lu']}_1": this_id_lu,
                        "relation": RELATION_NAME,
                        f"{CANDIDATE_FIELD_NAMES['id_lu']}_2": other_id_lu
                    })

        self.lu_relations = pd.DataFrame.from_records(relations).drop_duplicates()

    def drop_reciprocals(self):
        self.lu_relations = pd.concat(
            [
                self.lu_relations,
                self.swap_col_names(self.lu_relations,
                                    col_name_1=f"{CANDIDATE_FIELD_NAMES['id_lu']}_1",
                                    col_name_2=f"{CANDIDATE_FIELD_NAMES['id_lu']}_2")
            ]
        ).drop_duplicates()

    def add_relations_info(self):
        if self.entries is None:
            self.init_entries()

        self.lu_relations = self.lu_relations.merge(
            self.entries[RELATION_ENTRIES_FIELD_NAMES.values()].add_suffix("_1"),
            how="left",
            left_on=f"{CANDIDATE_FIELD_NAMES['id_lu']}_1",
            right_on=f"{RELATION_ENTRIES_FIELD_NAMES['idLu']}_1"
        ).merge(
            self.entries[RELATION_ENTRIES_FIELD_NAMES.values()].add_suffix("_2"),
            how="left",
            left_on=f"{CANDIDATE_FIELD_NAMES['id_lu']}_2",
            right_on=f"{RELATION_ENTRIES_FIELD_NAMES['idLu']}_2"
        )[[
            f"{CANDIDATE_FIELD_NAMES['id_lu']}_1",
            f"{RELATION_ENTRIES_FIELD_NAMES['word']}_1",
            f"{RELATION_ENTRIES_FIELD_NAMES['pos']}_1",
            f"{RELATION_ENTRIES_FIELD_NAMES['fnDefinition']}_1",
            RELATION_NAME,
            f"{CANDIDATE_FIELD_NAMES['id_lu']}_2",
            f"{RELATION_ENTRIES_FIELD_NAMES['word']}_2",
            f"{RELATION_ENTRIES_FIELD_NAMES['pos']}_2",
            f"{RELATION_ENTRIES_FIELD_NAMES['fnDefinition']}_2"
        ]].drop_duplicates()

    def drop_relations(self):
        if self.lu_relations is not None:
            self.lu_relations = self.lu_relations.loc[~self.lu_relations[RELATION_NAME].isin(self.RELATIONS_TO_DROP)]

    def write_to_file(self):
        if self.lu_relations is not None:
            self.lu_relations.to_csv(self.out_path)

    @staticmethod
    def swap_col_names(df: pd.DataFrame, col_name_1: str, col_name_2: str):
        df[[col_name_1, col_name_2]] = df[[col_name_2, col_name_1]]
        return df

    @staticmethod
    def convert_edges(edges_string: str) -> List[List[str]]:
        # rel, id
        split_edges = edges_string.split("||")
        split_edges = [el.split("|") for el in split_edges]
        return split_edges

    def process_data(self):
        self.get_relations()
        self.generate_output()
        self.add_relations_info()
        self.drop_relations()
        self.write_to_file()

