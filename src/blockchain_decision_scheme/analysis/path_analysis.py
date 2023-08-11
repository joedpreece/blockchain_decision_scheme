from typing import Dict

import pandas as pd

import blockchain_decision_scheme.decision_tree as dt


class DecisionTreePathAnalysis:

    def __init__(self, decision_tree_collection):
        self.decision_tree_collection = decision_tree_collection
        self.df = self.__create_paths_df()

    def __create_paths_df(self) -> pd.DataFrame:
        dfs = []

        for name, decision_tree in self.decision_tree_collection.decision_trees.items():
            dfs.append(decision_tree.create_paths_df())

        return pd.concat(dfs)

    def get_scheme_df(self, scheme_name: str) -> pd.DataFrame:
        return self.df[self.df[dt._column_names[0]] == scheme_name]

    def get_paths_count(self, scheme_name: str) -> int:
        return len(self.get_scheme_df(scheme_name=scheme_name))

    def get_path_length_mean(self, scheme_name: str) -> float:
        def list_length(lst):
            return len(lst)

        df = self.get_scheme_df(scheme_name=scheme_name)

        return df['Path'].apply(list_length).mean()

    def get_path_length_std(self, scheme_name: str) -> float:
        def list_length(lst):
            return len(lst)

        df = self.get_scheme_df(scheme_name=scheme_name)

        return df['Path'].apply(list_length).std()

    def get_outcome_value_count(self, scheme_name: str, value: str):
        df = self.get_scheme_df(scheme_name=scheme_name)
        df = df[df['Outcome'] == value]
        return len(df)