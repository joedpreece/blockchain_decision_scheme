import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import blockchain_decision_scheme.decision_tree as dt
import blockchain_decision_scheme.analysis.path_analysis as pa


def get_attribute_count(decision_tree: dt.DecisionTree, attribute: str):
    attribute_counts = decision_tree.df[dt._column_names[10]].value_counts()
    if attribute not in attribute_counts:
        return 0
    else:
        return attribute_counts[attribute]


def compute_structural_encoding(decision_tree: dt.DecisionTree, paths):

    df_decision = decision_tree.df[decision_tree.df[dt._column_names[1]] == 'Decision']
    df_outcome = decision_tree.df[decision_tree.df[dt._column_names[1]] == 'Outcome']

    return {
        'Decision Vertices: Count': len(df_decision),
        'Outcome Vertices: Count': len(df_outcome),
        'Paths: Count': paths.get_paths_count(scheme_name=decision_tree.name),
        'Paths: Avoid Count': paths.get_outcome_value_count(scheme_name=decision_tree.name, value='Avoid'),
        'Paths: Blockchain Count': paths.get_outcome_value_count(scheme_name=decision_tree.name, value='Blockchain'),
        'Paths: Average Length': paths.get_path_length_mean(scheme_name=decision_tree.name),
        'Paths: STD Length': paths.get_path_length_std(scheme_name=decision_tree.name)
    }


def normalise_column(column):
    scaler = MinMaxScaler()
    return pd.Series(scaler.fit_transform(column.values.reshape(-1, 1)).ravel())


def get_structural_similarity_matrix(decision_tree_collection: dt.DecisionTreeCollection, paths):
    dicts = {}
    for name, decision_tree in decision_tree_collection.decision_trees.items():
        dicts[name] = compute_structural_encoding(decision_tree, paths=paths)
    df = pd.DataFrame(dicts)
    df = df.transpose()

    normalised_df = df.apply(lambda column: normalise_column(column))

    similarity_matrix = cosine_similarity(df)
    similarity_matrix_df = pd.DataFrame(similarity_matrix, columns=df.index, index=df.index)

    return similarity_matrix * 100, similarity_matrix_df * 100


if __name__ == '__main__':
    decision_tree_collection = dt.DecisionTreeCollection()
    paths = pa.DecisionTreePathAnalysis(decision_tree_collection=decision_tree_collection)

    matrix = get_structural_similarity_matrix(decision_tree_collection=decision_tree_collection, paths=paths)

    # for i in range(0, 30):
    #     print(i, matrix[1].index[i])
