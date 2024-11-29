import numpy as np
import torch
from PIL import Image
import networkx as nx
from collections import Counter
import pandas as pd

class StreetView_Graph:
    def __init__(self, image_path):#应该是街景图像的语义分割结果文件的路径
        self.image_path = image_path
        self.svi_seg = None
        self.adjacency_dict = None
        self.adjacency_matrix = None
        self.weight_graph = None

    def load_image(self): #读取语义分割结果
        self.svi_seg = np.array(Image.open(self.image_path))
        return self.svi_seg

    def calculate_neighborhood(self):
        #计算图像中每个语义类别与其邻居类别的关系，生成邻接字典。
        #遍历语义分割图像的每个像素点，检查四个方向（上下左右）的邻居像素。如果邻居像素属于不同类别，将该邻居类别加入当前类别的邻接列表。
        #构建一个字典 self.adjacency_dict，键为语义类别，值为相邻类别的列表。
        if self.svi_seg is None:
            self.load_image()

        unique_labels = np.unique(self.svi_seg)
        self.adjacency_dict = {label: [] for label in unique_labels}

        for i in range(self.svi_seg.shape[0]):
            for j in range(self.svi_seg.shape[1]):
                label = self.svi_seg[i, j] #读取该点的语义类别
                neighbors = []
                
                if i > 0 and self.svi_seg[i - 1, j] != label:#如果点(i,j)与左侧点(i-1,j)属于不同类别，则记录左侧点(i-1,j)的类别
                    neighbors.append(self.svi_seg[i - 1, j])
                if i < self.svi_seg.shape[0] - 1 and self.svi_seg[i + 1, j] != label: #如果点(i,j)与右侧点(i+1,j)属于不同类别，则记录右侧点(i+1,j)的类别
                    neighbors.append(self.svi_seg[i + 1, j])
                if j > 0 and self.svi_seg[i, j - 1] != label:#如果点(i,j)与上边点(i,j-1)属于不同类别，则记录上边点(i,j-1)的类别
                    neighbors.append(self.svi_seg[i, j - 1])
                if j < self.svi_seg.shape[1] - 1 and self.svi_seg[i, j + 1] != label:#如果点(i,j)与下边点(i,j+1)属于不同类别，则记录上边点(i,j+1)的类别
                    neighbors.append(self.svi_seg[i, j + 1])
                self.adjacency_dict[label].extend(neighbors) #键为label的字典添加列表neighbors
        
        return self.adjacency_dict

    def generate_adjacency_matrix(self):
        #从邻接字典生成类别之间的邻接矩阵。
        #遍历邻接字典中的每个类别及其邻居列表，用 Counter 统计相邻类别的频次。
        # 构建一个 pandas.DataFrame 表示邻接矩阵。
        # 对邻接矩阵进行预处理：
        # 过滤掉像素占比低于 1% 的类别。
        # 确保矩阵的行列索引一致。
        # 返回值：邻接矩阵 self.adjacency_matrix。

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
        # 根据邻接矩阵构建一个带权无向图。
        if self.adjacency_matrix is None:
            self.generate_adjacency_matrix()
        # 统计每个语义类别的像素数，并将其作为节点的属性。
        category_counts = dict(zip(*np.unique(self.svi_seg, return_counts=True)))
        
        node_attribute_df = pd.DataFrame(category_counts, index=['count']).T.loc[self.adjacency_matrix.columns, :]
        node_attribute = [(i, {'count': attribute[0]}) for i, attribute in enumerate(node_attribute_df.values)]
        # 遍历邻接矩阵，将非零元素作为边的权重，构建图的边。
        self.weight_graph = nx.Graph()
        adjacency_matrix_arr = np.array(self.adjacency_matrix)
        self.weight_graph.add_nodes_from(node_attribute)

        rows, cols = np.where(adjacency_matrix_arr != 0)
        weights = adjacency_matrix_arr[rows, cols]
        edges = zip(rows, cols, weights)
        self.weight_graph.add_weighted_edges_from(edges)

        return self.weight_graph

    def calculate_embedding(self):
        # 计算图的嵌入表示。
        if self.weight_graph is None:
            self.create_svi_graph()
        # 当前实现直接使用邻接矩阵的数值表示作为嵌入（简化处理）。
        # Use node2vec or another graph embedding method here
        # 可以扩展为使用更复杂的图嵌入方法（如 Node2Vec）。
        # For simplicity, we'll use the adjacency matrix as a basic embedding
        embedding = torch.FloatTensor(self.adjacency_matrix.values)
        return embedding

def calculate_adjacency_and_embedding(image_path):
    #从语义分割图像中计算邻接关系并生成图嵌入。
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
