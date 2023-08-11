import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import blockchain_decision_scheme.decision_tree as dt


def get_semantic_similarity_matrix(decision_tree_collection: dt.DecisionTreeCollection):
    # Collect the model encodings into a dataframe
    encodings = {}
    for name, decision_tree in decision_tree_collection.decision_trees.items():
        encodings[name] = decision_tree.compute_semantic_encoding()
    df = pd.DataFrame(encodings)

    # The dataframe needs to be transposed for the cosine similarity function
    df = df.transpose()

    similarity_matrix = cosine_similarity(df)
    similarity_matrix_df = pd.DataFrame(similarity_matrix, columns=df.index, index=df.index)

    return similarity_matrix * 100, similarity_matrix_df * 100


if __name__ == '__main__':
    decision_tree_collection = dt.DecisionTreeCollection()
    matrix = get_semantic_similarity_matrix(decision_tree_collection=decision_tree_collection)