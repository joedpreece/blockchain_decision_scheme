from pathlib import Path
import importlib.resources as pkg_resources
from typing import Dict, List

import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

default_filepath = pkg_resources.path('blockchain_decision_scheme', 'Input.csv')
gourisetti_filepath = pkg_resources.path('blockchain_decision_scheme', 'gourisetti.csv')

_column_names = {
    0: 'Blockchain Decision Scheme',
    1: 'Vertex Type',
    2: 'Vertex ID',
    3: 'Vertex Label',
    4: 'Decision A Label',
    5: 'Decision A Next Vertex ID',
    6: 'Decision B Label',
    7: 'Decision B Next Vertex ID',
    8: 'Decision C Label',
    9: 'Decision C Next Vertex ID',
    10: 'Attribute',
    11: 'Encodings'
}


class DecisionTree:

    def __init__(self,
                 name: str,
                 df: pd.DataFrame):
        """
        Creates a new decision tree.

        .. csv-table:: Table 1: The required format for a decision tree input dataframe.
            :header: Type, Vertex ID, Question, Decision 1, Decision 1 Next, Decision 2, Decision 2 Next, Decision 3, Decision 3 Next

            Decision,1,Do we need a database?,Yes,2,No,A,,
            Decision,2,Does it require shared write access?,Yes,3,No,A,,
            Decision,3,Are the writers known and trusted?,Yes,4,No,5,,
            Decision,4,Can you align the writers' incentives?,Yes,B3,No,A,,
            Decision,5,Do we need to trust third parties?,Yes,6,No,7,,
            Decision,6,Do we control the consensus?,Yes,B3,No,B2,,
            Decision,7,Do you want transactions to be public?,Yes,B1,No,6,,
            Outcome,A,We don't need blockchain,,,,,,
            Outcome,B1,We can use a public blockchain,,,,,,
            Outcome,B2,We can use a hybrid blockchain,,,,,,
            Outcome,B3,We can use a private blockchain,,,,,,

        :param name: The name of the decision tree model.
        :param df: The associated dataframe.
        """

        self.name = name
        """
        The name of the decision tree model.
        """

        self.df = df
        """
        The dataframe representation of the decision tree.
        """

        self.digraph = nx.DiGraph()
        """
        The graph representation of the decision tree.
        """

        self.digraph_decision_vertices = None
        """
        The subgraph with just decision vertices.
        """

        self.digraph_outcome_vertices = None
        """
        The subgraph with just outcome vertices.
        """

        self.semantic_encoding = None
        """
        The encoding of the model label, used to compare language similarity with other models.
        """

        # Compute the digraph from the df information.
        self.__compute_digraph()

    def __compute_digraph(self):

        # Make temporary filters of the df for the decision and outcome vertices.
        df_decision = self.df[self.df[_column_names[1]] == 'Decision']
        df_outcome = self.df[self.df[_column_names[1]] == 'Outcome']

        self.__generate_decision_vertices(df_decision=df_decision)
        self.__generate_outcome_vertices(df_outcome=df_outcome)
        self.__generate_edges(df_decision=df_decision)

        self.digraph_decision_vertices = [x for x, y in self.digraph.nodes(data=True) if y['type'] == 'Decision']
        self.digraph_outcome_vertices = [x for x, y in self.digraph.nodes(data=True) if y['type'] == 'Outcome']

    def __generate_decision_vertices(self, df_decision: pd.DataFrame):
        for index, row in df_decision.iterrows():
            self.digraph.add_node(row[_column_names[2]],
                                  type='Decision',
                                  attribute=row[_column_names[10]],
                                  label=row[_column_names[3]],
                                  )

    def __generate_outcome_vertices(self, df_outcome: pd.DataFrame):
        for index, row in df_outcome.iterrows():
            self.digraph.add_node(row[_column_names[2]],
                                  type='Outcome',
                                  classification=row[_column_names[2]],
                                  label=row[_column_names[3]])

    def __generate_edges(self, df_decision: pd.DataFrame):
        for index, row in df_decision.iterrows():
            self.digraph.add_edge(row[_column_names[2]], row[_column_names[5]], label=row[_column_names[4]])
            self.digraph.add_edge(row[_column_names[2]], row[_column_names[7]], label=row[_column_names[6]])

            if not pd.isna(row[_column_names[8]]):
                self.digraph.add_edge(row[_column_names[2]], str(row[_column_names[9]]), label=row[_column_names[8]])

    def compute_semantic_encoding(self) -> List:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        df_decision = self.df[self.df[_column_names[1]] == 'Decision']
        model_str = ' '.join(df_decision[_column_names[3]].tolist())
        return model.encode(model_str)

    def compute_structure(self):
        df_paths = self.create_paths_df()

        return {'Name': self.name,
                'Number of Decision Vertices': len(self.digraph_decision_vertices),
                'Number of Outcome Vertices': len(self.digraph_outcome_vertices),
                'Number of Paths': len(df_paths)
                }

    def get_all_paths(self) -> List:
        """
        Computes all paths between the first Decision vertex and all possible Outcome vertices.
        :return: A list of the paths, with paths represented as the sequence of Vertex IDs.
        :rtype List:
        """
        paths = []
        for target in self.digraph_outcome_vertices:
            all_simple_paths = nx.all_simple_paths(G=self.digraph, source='1', target=target)
            for path in all_simple_paths:
                paths.append(path)
        return paths

    def create_paths_df(self) -> pd.DataFrame:

        df = pd.DataFrame({
            'Path': self.get_all_paths()
        })

        df[_column_names[0]] = self.name

        df['Length'] = df.apply(lambda row: len(row['Path']), axis=1)

        df['Outcome'] = df.apply(lambda row: self.__determine_outcome_high(row), axis=1)
        df['Outcome Detailed'] = df.apply(lambda row: self.__determine_outcome_medium(row), axis=1)
        return df

    @staticmethod
    def __determine_outcome_high(row) -> str:
        outcome = row['Path'][-1]
        if 'A' in outcome:
            return 'Avoid'
        if 'B' in outcome:
            return 'Blockchain'

    @staticmethod
    def __determine_outcome_medium(row) -> str:
        outcome = row['Path'][-1]
        if 'A' in outcome:
            if 'A1' in outcome:
                return 'Use a Database'
            else:
                return 'Avoid Blockchain'
        if 'B' in outcome:
            if 'B1' in outcome:
                return 'Use Public Blockchain'
            if 'B2' in outcome:
                return 'Use Hybrid Blockchain'
            if 'B3' in outcome:
                return 'Use Private Blockchain'
            else:
                return 'Use Blockchain'


class DecisionTreeCollection:

    def __init__(self,
                 filepath: str = default_filepath):
        """
        Creates a bank of decision tree based blockchain decision models (BDSs).

        It ingests the decision trees from a CSV file. This package requires the file to be of a certain format, with the following headers:

        - **Model** The name of the model e.g. 'Birch'
        - **Type** The type of vertex. Must be either 'Decision' or 'Outcome'.
        - **Vertex ID** The id of the vertex. Convention follows that decision vertices are numbered from 1 based upon their ordering, whilst the Outcome vertices are labelled as follows:

            - **A** Avoid Blockchain

                - **A1** Database
                - **A2** Spreadsheet

            - **B** Use Blockchain

                - **B1** Public Blockchain
                - **B2** Hybrid Blockchain
                - **B3** Private Blockchain

        - **Vertex Label** The label for the vertex. For Decision vertices, this is usually the question asked.
        - **Decision 1 Label** Provides the first of the possible decisions (usually one of 'yes'/'no'). Only required for Decision vertices. For the first option,
        - **Decision 1 Next Vertex ID** Specifies the Vertex ID of the next vertex in the decision tree. Only required for Decision vertices.
        - **Decision 2 Label** Provides the second of the possible decisions (usually one of 'yes'/'no'). Only required for Decision vertices. For the first option,
        - **Decision 2 Next Vertex ID** Specifies the Vertex ID of the next vertex in the decision tree. Only required for Decision vertices.
        - **Decision 3 Label** Provides the third of the possible decisions, if necessary. Only required for Decision vertices. For the first option,
        - **Decision 3 Next Vertex ID** Specifies the Vertex ID of the next vertex in the decision tree. Only required for Decision vertices.

        To see an example, please load the default DecisionTreeCollection and access the df, which is loaded directly from the default file.

        :param filepath: The path to the CSV file. If empty, the default decision trees are loaded.
        :type filepath: str:
        """

        self.df = None
        """
        The dataframe representing the decision trees.
        """
        self.decision_trees = {}
        """
        A dictionary of the decision trees. Keys are strings of the model names, and values are the corresponding DecisionTree.
        """

        # Open the filepath.
        with open(filepath, 'r') as file:
            # Extract the CSV as a dataframe.
            self.df = pd.read_csv(filepath_or_buffer=file, dtype=str)
            self.__classify_labels()
            self.__build_decision_trees()

    def __build_decision_trees(self):
        """
        Creates all the decision trees from the CSV file.
        :return:
        """
        model_names = self.df[_column_names[0]].unique()
        for model_name in model_names:
            model_df = self.df[self.df[_column_names[0]] == model_name]
            model_df = model_df.drop(columns=[_column_names[0]])
            decision_tree = DecisionTree(name=model_name, df=model_df)
            self.decision_trees[model_name] = decision_tree

    def __classify_labels(self):
        df_gourisetti = pd.read_csv(gourisetti_filepath)
        df_gourisetti = df_gourisetti[['Domain', 'Question']]

        model = SentenceTransformer('bert-base-nli-mean-tokens')

        # Get the model encodings for the labels.
        model_encodings = model.encode(self.df[_column_names[3]].tolist())
        gourisetti_encodings = model.encode(df_gourisetti['Question'].tolist())

        # Add the encodings to the df for future usage. We need to reshape the array to do this.
        # new_col = np.reshape(model_encodings, (361, 1, 768))
        # self.df[_column_names[11]] = pd.Series(new_col[:,0,:].tolist())

        # Compute the similarities between the models and Gourisetti encodings.
        similarity_matrix = cosine_similarity(model_encodings, gourisetti_encodings)
        highest_similarity_indices = np.argmax(similarity_matrix, axis=1)
        self.df[_column_names[10]] = df_gourisetti.iloc[highest_similarity_indices]['Domain'].tolist()

    def get_decision_tree(self,
                          model: str) -> DecisionTree:
        """
        Gets a decision tree from the collection.
        :param model: The name of the decision tree model.
        :return: The decision tree.
        """
        return self.decision_trees[model]

