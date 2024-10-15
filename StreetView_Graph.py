import numpy as np
import torch
from PIL import Image
import networkx as nx
from collections import Counter
import pandas as pd

class StreetView_Graph:
    def __init__(self, image_path):
        self.image_path = image_path
        self.svi_seg = None
        self.adjacency_dict = None
        self.adjacency_matrix = None
        self.weight_graph = None

    def load_image(self):
        self.svi_seg = np.array(Image.open(self.image_path))
        return self.svi_seg

    def calculate_neighborhood(self):
        if self.svi_seg is None:
            self.load_image()

        unique_labels = np.unique(self.svi_seg)
        self.adjacency_dict = {label: [] for label in unique_labels}

        for i in range(self.svi_seg.shape[0]):
            for j in range(self.svi_seg.shape[1]):
                label = self.svi_seg[i, j]
                neighbors = []
                if i > 0 and self.svi_seg[i - 1, j] != label:
                    neighbors.append(self.svi_seg[i - 1, j])
                if i < self.svi_seg.shape[0] - 1 and self.svi_seg[i + 1, j] != label:
                    neighbors.append(self.svi_seg[i + 1, j])
                if j > 0 and self.svi_seg[i, j - 1] != label:
                    neighbors.append(self.svi_seg[i, j - 1])
                if j < self.svi_seg.shape[1] - 1 and self.svi_seg[i, j + 1] != label:
                    neighbors.append(self.svi_seg[i, j + 1])
                self.adjacency_dict[label].extend(neighbors)
        
        return self.adjacency_dict

    def generate_adjacency_matrix(self):
        if self.adjacency_dict is None:
            self.calculate_neighborhood()

        self.adjacency_matrix = pd.DataFrame()
        for i, cate in enumerate(self.adjacency_dict.keys()):
            if i == 0:
                self.adjacency_matrix = pd.DataFrame(Counter(self.adjacency_dict[cate]), index=[cate])
            else:
                self.adjacency_matrix = pd.concat([self.adjacency_matrix, pd.DataFrame(Counter(self.adjacency_dict[cate]), index=[cate])])
        
        self.adjacency_matrix.fillna(0, inplace=True)
        
        # Filter out less significant categories (e.g., less than 1% of total pixels)
        select_index = self.adjacency_matrix[self.adjacency_matrix.sum(axis=1) / self.adjacency_matrix.sum(axis=1).sum() > 0.01].index
        self.adjacency_matrix = self.adjacency_matrix.loc[select_index, select_index]
        self.adjacency_matrix = self.adjacency_matrix.reindex(columns=self.adjacency_matrix.index)

        return self.adjacency_matrix

    def create_svi_graph(self):
        if self.adjacency_matrix is None:
            self.generate_adjacency_matrix()

        category_counts = dict(zip(*np.unique(self.svi_seg, return_counts=True)))
        
        node_attribute_df = pd.DataFrame(category_counts, index=['count']).T.loc[self.adjacency_matrix.columns, :]
        node_attribute = [(i, {'count': attribute[0]}) for i, attribute in enumerate(node_attribute_df.values)]

        self.weight_graph = nx.Graph()
        adjacency_matrix_arr = np.array(self.adjacency_matrix)
        self.weight_graph.add_nodes_from(node_attribute)

        rows, cols = np.where(adjacency_matrix_arr != 0)
        weights = adjacency_matrix_arr[rows, cols]
        edges = zip(rows, cols, weights)
        self.weight_graph.add_weighted_edges_from(edges)

        return self.weight_graph

    def calculate_embedding(self):
        if self.weight_graph is None:
            self.create_svi_graph()

        # Use node2vec or another graph embedding method here
        # For simplicity, we'll use the adjacency matrix as a basic embedding
        embedding = torch.FloatTensor(self.adjacency_matrix.values)
        return embedding

def calculate_adjacency_and_embedding(image_path):
    """
    Calculate adjacency relationship based on the segmented image and obtain weighted initial values.
    
    :param image_path: Path to the segmented image file
    :return: Embedding (weighted initial values)
    """
    
    graph = StreetView_Graph(image_path)
    graph.load_image()
    graph.calculate_neighborhood()
    graph.generate_adjacency_matrix()
    graph.create_svi_graph()
    embedding = graph.calculate_embedding()
    
    return embedding

# Example usage:
if __name__ == "__main__":
    # Replace with your actual image path
    example_image_path = "path/to/your/segmented_image.jpg"
    
    embedding = calculate_adjacency_and_embedding(example_image_path)
    print("Embedding shape:", embedding.shape)
    print("Embedding:", embedding)
