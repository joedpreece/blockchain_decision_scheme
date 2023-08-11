import unittest

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datasets import Dataset
import evaluate

from nltk.corpus import stopwords

import re

import blockchain_decision_scheme.decision_tree as dt
import blockchain_decision_scheme.analysis.path_analysis as dta
import seaborn as sns
from transformers import AutoTokenizer, DataCollatorWithPadding, BertModel, AutoModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from blockchain_decision_scheme.analysis.combined_similarity import get_combined_similarity_matrix
from blockchain_decision_scheme.analysis.semantic_similarity import get_semantic_similarity_matrix
from blockchain_decision_scheme.analysis.structural_similarity import get_structural_similarity_matrix

run_vertex_analysis = True
run_path_analysis = False
run_similarity_analysis = False

figsize = (10, 15)
gridspec_kw = {'height_ratios': [15, 3]}

import matplotlib.colors as colors

# Define the colors you want in the colormap
colors_list = ['#6fa8dc', '#E06666']
# Create a colormap using LinearSegmentedColormap
cmap = colors.LinearSegmentedColormap.from_list('white_to_red', colors_list)


@unittest.skipIf(run_vertex_analysis is False, 'Vertex Analysis Skipped')
class TestVertexAnalysis(unittest.TestCase):
    i = 0

    def setUp(self):
        self.decision_trees = dt.DecisionTreeCollection()
        self.df = self.decision_trees.df
        self.df.replace(to_replace='Decision', value='Question', inplace=True)

    # @unittest.skip('Just skipping')
    def test_vertex_count(self):
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=figsize, gridspec_kw=gridspec_kw)
        colour_palette = sns.color_palette(['#e06666', '#6fa8dc'])

        sns.countplot(ax=ax0, data=self.df, y=dt._column_names[0], hue=dt._column_names[1], palette=colour_palette,
                      saturation=1)
        ax0.set_xlabel('')
        ax0.set_ylabel('Blockchain Decision Scheme')
        ax0.set_xlim([0, 15])

        df = self.df.groupby([dt._column_names[0], dt._column_names[1]]).count()
        df = df.reset_index()
        df = df.sort_values(by=dt._column_names[1], ascending=False)
        print(df)
        sns.boxplot(ax=ax1, data=df, x='Vertex ID', y='Vertex Type', palette=colour_palette, saturation=1)
        # plt.xlabel('Vertex Count')
        # plt.ylabel('Model')
        ax1.set_xlabel('Vertex Count')
        ax1.set_ylabel('Vertex Type')
        ax1.set_xlim([0, 15])

    @unittest.skip('Just skipping')
    def test_decision_vertex_domain_composition(self):
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=figsize, gridspec_kw=gridspec_kw)

        result = self.df.groupby(dt._column_names[0])[dt._column_names[10]].value_counts(normalize=True).mul(
            100).rename('Percent').reset_index()
        pivoted_df = result.pivot(index=dt._column_names[0], columns=dt._column_names[10],
                                  values='Percent').reset_index()
        pivoted_df.fillna(0, inplace=True)
        pivoted_df = pivoted_df.sort_values(by=dt._column_names[0], ascending=False)
        pivoted_df.plot(ax=ax0, kind='barh', x=dt._column_names[0], legend=False, stacked=True,
                        color=[
                            '#e06666',
                            '#f6b26b',
                            '#ffd966',
                            '#6fa8dc',
                            '#8e7cc3'
                        ])
        # plt.xlabel('Domain (%)')
        # plt.ylabel('Model')
        ax0.set_xlabel('Domain (%)')
        ax0.set_ylabel('Blockchain Decision Scheme')

        result = self.df[dt._column_names[10]].value_counts(normalize=True).mul(100).rename('Percent')
        result.plot.pie(ax=ax1, y='Percent', autopct='%1.1f%%',
                        colors=[
                            '#e06666',
                            '#6fa8dc',
                            '#ffd966',
                            '#f6b26b',
                            '#8e7cc3'
                        ])
        ax1.set_xlabel('')
        ax1.set_ylabel('')

    @unittest.skip('Just skipping')
    def test_decision_vertices_wordcloud(self):
        def clean_text(text):
            # Lowercase the text
            text = text.lower()

            # Remove special characters and numbers
            text = re.sub("[^a-zA-Z]", " ", text)

            # Tokenize the text
            words = nltk.word_tokenize(text)

            # Remove the stopwords
            words = [word for word in words if word not in stopwords.words("english")]

            # Join the words back into a single string
            cleaned_text = " ".join(words)

            return cleaned_text

        nltk.download('stopwords')
        from nltk.corpus import stopwords

        # Clean the 'Text' column
        self.df['cleaned_text'] = self.df[dt._column_names[3]].apply(clean_text)

        # Create a single string from the cleaned text
        cleaned_text = " ".join(self.df['cleaned_text'].tolist())

        # Generate the wordcloud
        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=cmap).generate(cleaned_text)

        # Display the wordcloud using matplotlib
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

    def tearDown(self) -> None:
        plt.tight_layout()
        plt.show()
        # plt.savefig((str(self.id()) + '.pdf'))
        plt.clf()


@unittest.skipIf(run_path_analysis is False, 'Path Analysis Skipped')
class TestPathAnalysis(unittest.TestCase):

    def setUp(self):
        self.decision_tree_collection = dt.DecisionTreeCollection()
        self.decision_tree_path = dta.DecisionTreePathAnalysis(decision_tree_collection=self.decision_tree_collection)
        self.df = self.decision_tree_path.df

    # @unittest.skip('Skipping')
    def test_path_counts(self):
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=figsize, gridspec_kw=gridspec_kw)

        df = self.df.groupby(['Blockchain Decision Scheme']).count()

        sns.countplot(ax=ax0, data=self.df, y='Blockchain Decision Scheme', color='#e06666', saturation=1)
        sns.boxplot(ax=ax1, data=df, x='Path', color='#e06666', saturation=1)

        ax0.set_xlabel('')
        ax0.set_ylabel('Blockchain Decision Scheme')
        ax0.set_xlim([0, 45])
        ax1.set_xlabel('Path Count')
        ax1.set_ylabel('')
        ax1.set_xlim([0, 45])

    # @unittest.skip('Skipping')
    def test_path_lengths(self):
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=figsize, gridspec_kw=gridspec_kw)

        sns.boxplot(ax=ax0, data=self.df, x='Length', y='Blockchain Decision Scheme', color='#e06666', saturation=1)
        sns.boxplot(ax=ax1, data=self.df, x='Length', color='#e06666', saturation=1)

        ax0.set_xlabel('')
        ax0.set_ylabel('Blockchain Decision Scheme')
        ax0.set_xlim([1, 14])
        ax1.set_xlabel('Path Length')
        ax1.set_ylabel('')
        ax1.set_xlim([1, 14])

    # @unittest.skip('Skipping')
    def test_path_outcomes(self):
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=figsize, gridspec_kw=gridspec_kw)
        self.df['A'] = 'All'
        colour_palette = sns.color_palette(['#e06666', '#6fa8dc'])
        sns.violinplot(ax=ax0, data=self.df, x='Length', y='Blockchain Decision Scheme', hue='Outcome', split=True,
                       palette=colour_palette, saturation=1)
        sns.violinplot(ax=ax1, data=self.df, x='Length', y='A', hue='Outcome', split=True, palette=colour_palette, saturation=1)

        ax0.set_xlabel('')
        ax0.set_ylabel('Blockchain Decision Scheme')
        ax0.set_xlim([0, 14])
        ax1.set_xlabel('Path Length')
        ax1.set_ylabel('')
        ax1.set_xlim([0, 14])
        # ax0.legend().remove()
        # ax1.legend().remove()

    def test_path_outcomes_detailed(self):
        normalise = True

        plot_df = self.df.groupby(['Blockchain Decision Scheme'])['Outcome Detailed'].value_counts(normalize=normalise)
        plot_df = plot_df.to_frame()

        if normalise:
            # plot_df = plot_df.rename(columns={'Outcome Detailed': 'Value'})
            plot_df['proportion'] = plot_df['proportion'].mul(100)
        plot_df = plot_df.reset_index()

        plot_df = plot_df.pivot_table(values='proportion', index='Blockchain Decision Scheme',
                                      columns='Outcome Detailed', aggfunc='first')
        plot_df = plot_df[
            ['Avoid Blockchain', 'Use a Database', 'Use Blockchain', 'Use Private Blockchain', 'Use Hybrid Blockchain',
             'Use Public Blockchain']]

        plot_df = plot_df.sort_values(by='Blockchain Decision Scheme', ascending=False)

        plot_df.fillna(0, inplace=True)

        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=figsize, gridspec_kw=gridspec_kw)

        plot_df.plot(ax=ax0, kind='barh', stacked='true',
                     color=['#e06666', '#e06666', '#6fa8dc', '#6fa8dc', '#6fa8dc', '#6fa8dc'])
        bars = ax0.patches
        patterns = ('N', '**', 'N', '..', 'oo', 'OO', 'o', '\\', '\\\\')
        hatches = [p for p in patterns for i in range(len(plot_df))]
        for bar, hatch in zip(bars, hatches):
            if hatch != 'N':
                bar.set_hatch(hatch)

        ax0.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')

        # Get the average
        average = plot_df.mean()
        average = average.to_frame()
        average = average.transpose()

        average = average.rename(index={0: 'Average'})

        average.plot(ax=ax1, kind='barh', stacked='true',
                     color=['#e06666', '#e06666', '#6fa8dc', '#6fa8dc', '#6fa8dc', '#6fa8dc'])
        bars = ax1.patches
        patterns = ('N', '**', 'N', '..', 'oo', 'OO', 'o', '\\', '\\\\')
        hatches = [p for p in patterns for i in range(len(average))]
        for bar, hatch in zip(bars, hatches):
            if hatch != 'N':
                bar.set_hatch(hatch)

        ax0.set_xlabel('')
        ax0.set_ylabel('Blockchain Decision Scheme')
        # ax0.set_xlim([0, 14])
        ax1.set_xlabel('Outcome Composition (%)')
        ax1.set_ylabel('')
        # ax1.set_xlim([0, 14])
        # ax0.legend().remove()
        ax1.legend().remove()

    def tearDown(self) -> None:
        plt.tight_layout()
        # plt.show()
        plt.savefig((str(self.id()) + '.pdf'))
        plt.clf()


@unittest.skipIf(run_similarity_analysis is False, 'Similarity Analysis Skipped')
class TestModelSimilarityAnalysis(unittest.TestCase):

    def setUp(self):
        self.decision_tree_collection = dt.DecisionTreeCollection()

    # @unittest.skip('Skip')
    def test_structural_similarity(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        self.decision_tree_path = dta.DecisionTreePathAnalysis(decision_tree_collection=self.decision_tree_collection)
        self.matrix = get_structural_similarity_matrix(decision_tree_collection=self.decision_tree_collection,
                                                       paths=self.decision_tree_path)
        sns.heatmap(self.matrix[1],
                    annot=True,
                    cmap=cmap,
                    # linewidths=.5,
                    fmt=".0f",)

    @unittest.skip('Skip')
    def test_semantic_similarity(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        self.matrix = get_semantic_similarity_matrix(decision_tree_collection=self.decision_tree_collection)
        sns.heatmap(self.matrix[1],
                    annot=True,
                    cmap=cmap,
                    # linewidths=.5,
                    fmt=".0f")

    @unittest.skip('Skip')
    def test_combined_similarity(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        self.matrix = get_combined_similarity_matrix(decision_tree_collection=self.decision_tree_collection)
        sns.heatmap(self.matrix[1],
                    annot=True,
                    cmap=cmap,
                    # linewidths=.5,
                    fmt=".0f")

    @unittest.skip('Skip')
    def test_combined_similarity_top_ten(self):
        self.matrix = get_combined_similarity_matrix(decision_tree_collection=self.decision_tree_collection)
        array = self.matrix[0]
        df = self.matrix[1]

        # Step 2: Flatten the upper triangle of the similarity matrix
        upper_triangle = np.triu(df, k=1).flatten()

        # Step 3: Sort the flattened similarity scores in descending order
        sorted_scores = np.sort(upper_triangle)[::-1]

        # Step 4: Select the top ten pairs
        top_ten_scores = sorted_scores[:10]

        # Find the indices of the top ten scores in the flattened matrix
        top_ten_indices = np.argpartition(upper_triangle, -10)[-10:]

        # Get the corresponding indices in the original matrix
        n = array.shape[0]
        top_ten_indices = [(i // n, i % n) for i in top_ten_indices]

        # Create a DataFrame to display the top ten pairs and scores
        top_ten_pairs = pd.DataFrame(top_ten_indices, columns=['Object 1', 'Object 2'])
        top_ten_pairs['Similarity Score'] = top_ten_scores

        # Replace the indices with labels
        top_ten_pairs = top_ten_pairs.replace({'Object 1': df.index, 'Object 2': df.index})

        # Print the top ten pairs with labels
        print(top_ten_pairs)

    def tearDown(self) -> None:
        # plt.tight_layout()
        plt.show()
        # plt.savefig((str(self.id()) + '.pdf'))
        plt.clf()


if __name__ == '__main__':
    unittest.main()
