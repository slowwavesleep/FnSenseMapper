from ranker import CandidateRanker
from relation_generator import RelationGenerator

if __name__ == "__main__":
    ranker = CandidateRanker(data_path="resources/bn_entries.csv",
                             out_path="resources/candidates.jsonl",
                             no_header=True,
                             model_name="sentence-transformers/LaBSE",
                             n_candidates=3,
                             substring_condition=True,
                             cutoff_similarity=0.3)

    ranker.write_candidates()

    relations_gen = RelationGenerator(edges_path="resources/bn_edges.csv",
                                      candidates_path="resources/candidates.jsonl",
                                      entries_path="resources/bn_entries.csv",
                                      no_header=True,
                                      out_path="resources/qualia_relations.csv")

    relations_gen.process_data()




