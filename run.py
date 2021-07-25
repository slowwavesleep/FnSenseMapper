import argparse

from yaml import safe_load

from ranker import CandidateRanker
from relation_generator import RelationGenerator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run FN sense mapping with specified settings"
    )
    parser.add_argument(
        dest="config",
        nargs="?",
        type=str,
        help="Path to config file",
        default="config.yml",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = safe_load(file)

    data_path = config["paths"]["data_path"]
    edges_path = config["paths"]["edges_path"]
    ranker_out_path = config["paths"]["ranker_out_path"]
    relations_out_path = config["paths"]["relations_out_path"]

    do_ranking = config.get("do_ranking", True)
    no_header = config.get("no_header", True)
    substring_condition = config.get("substring_condition", True)

    embedder_model = config["embedder_model"]

    cutoff_similarity = config["cutoff_similarity"]
    n_candidates = config.get("n_candidates", 1)

    if do_ranking:
        ranker = CandidateRanker(
            data_path=data_path,
            out_path=ranker_out_path,
            no_header=no_header,
            model_name=embedder_model,
            n_candidates=n_candidates,
            substring_condition=substring_condition,
            cutoff_similarity=cutoff_similarity,
        )

        ranker.write_candidates()

    relations_gen = RelationGenerator(
        edges_path=edges_path,
        candidates_path=ranker_out_path,
        entries_path=data_path,
        no_header=no_header,
        out_path=relations_out_path,
    )

    relations_gen.process_data()
