from ranker import CandidateRanker
from relation_generator import RelationGenerator

if __name__ == "__main__":

    DO_RANKING = True
    DATA_PATH = "resources/bn_entries.csv"
    NO_HEADER = True
    SUBSTRING_CONDITION = True
    N_CANDIDATES = 3
    RANKER_OUT_PATH = "resources/candidates.jsonl"
    RELATIONS_OUT_PATH = "resources/qualia_relations.csv"
    EDGES_PATH = "resources/bn_edges.csv"
    MODEL_NAME = "sentence-transformers/LaBSE"
    CUTOFF_SIMILARITY = 0.3

    if DO_RANKING:
        ranker = CandidateRanker(data_path=DATA_PATH,
                                 out_path=RANKER_OUT_PATH,
                                 no_header=NO_HEADER,
                                 model_name=MODEL_NAME,
                                 n_candidates=N_CANDIDATES,
                                 substring_condition=SUBSTRING_CONDITION,
                                 cutoff_similarity=CUTOFF_SIMILARITY)

        ranker.write_candidates()

    relations_gen = RelationGenerator(edges_path=EDGES_PATH,
                                      candidates_path=RANKER_OUT_PATH,
                                      entries_path=DATA_PATH,
                                      no_header=NO_HEADER,
                                      out_path=RELATIONS_OUT_PATH)

    relations_gen.process_data()




