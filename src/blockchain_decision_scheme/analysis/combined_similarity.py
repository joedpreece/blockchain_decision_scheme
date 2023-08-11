import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import blockchain_decision_scheme.decision_tree as dt
from blockchain_decision_scheme.analysis.path_analysis import DecisionTreePathAnalysis
from blockchain_decision_scheme.analysis.semantic_similarity import get_semantic_similarity_matrix
from blockchain_decision_scheme.analysis.structural_similarity import get_structural_similarity_matrix


def get_combined_similarity_matrix(decision_tree_collection: dt.DecisionTreeCollection):

    decision_tree_path_analysis = DecisionTreePathAnalysis(decision_tree_collection=decision_tree_collection)

    str, str_df = get_structural_similarity_matrix(decision_tree_collection=decision_tree_collection, paths=decision_tree_path_analysis)
    sem, sem_df = get_semantic_similarity_matrix(decision_tree_collection=decision_tree_collection)

    df = (sem_df + str_df) / 2

    matrix = (sem + str) / 2

    return matrix, df


if __name__ == '__main__':
    decision_tree_collection = dt.DecisionTreeCollection()
    matrix = get_combined_similarity_matrix(decision_tree_collection=decision_tree_collection)