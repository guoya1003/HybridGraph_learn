import math
import numpy as np
from torch_geometric.nn import GATConv
import torch, os
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GATConv, TopKPooling
import torch.nn as nn
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# 动态自然断点分类
def dynamic_natural_breaks_classification(data, n):
    data_without_none = [x for x in data if x is not None]
    breakpoints = [min(data_without_none) + i * (max(data_without_none) - min(data_without_none)) / n for i in range(1, n)]
    categories = [-1 if x is None else next((i for i, b in enumerate(breakpoints) if x < b), n - 1) for x in data]
    return categories

# 等量分位数分类
def equal_quantile_classification(data, n):
    data_without_none = [x for x in data if x is not None]
    quantiles = [np.quantile(data_without_none, q=i / n) for i in range(1, n)]
    categories = [-1 if x is None else next((i for i, q in enumerate(quantiles) if x < q), n - 1) for x in data]
    return categories

# 生成虚拟数据的函数
# 修改生成虚拟数据的函数，以使用 split_ratio
def generate_dummy_graph_data(num_nodes=100, num_node_features=19, num_classes=3, split_ratio=0.8):
    x = torch.rand((num_nodes, num_node_features))
    y = torch.randint(0, num_classes, (num_nodes,))

    #spatial dependent edge_index
    edge_index1 = torch.randint(0, num_nodes, (2, num_nodes * 2))
    data1 = Data(x=x, edge_index=edge_index1, y=y)
    #spatial interaction edge_index
    edge_index2 = torch.randint(0, num_nodes, (2, num_nodes * 2))
    data2 = Data(x=x, edge_index=edge_index2, y=y)

    # 根据 split_ratio 分配训练和测试掩码
    num_train = int(num_nodes * split_ratio)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask[num_train:] = True
    data1.train_mask = train_mask
    data1.test_mask = test_mask

    data2.train_mask = train_mask
    data2.train_mask = train_mask

    return data1,data2

# 计算指标
def return_score(y_test, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Classification Accuracy: {accuracy}")
    # Calculate precision
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    # print(f"Precision: {precision}")
    # Calculate recall
    recall = recall_score(y_test, y_pred, average='weighted')
    # print(f"Recall: {recall}")
    return accuracy, precision, recall

# 对比实验函数 Comparison Experiment
def compare_method(X_train, X_test, y_train, y_test):
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(),
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Neural Network': MLPClassifier(),
        # Add more classifiers as needed
    }
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Evaluate the classifier and store results
        accuracy, precision, recall = return_score(y_test, y_pred)
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}
    print("Comparison Experiment:",results)
    return results

def train(mixed_data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(mixed_data)  # Perform a single forward pass.

    train_mask = mixed_data[0].train_mask
    test_mask= mixed_data[0].test_mask
    y = mixed_data[0].y

    loss = criterion(out[train_mask],
                     y[train_mask])  # Compute the loss solely based on the training nodes.
    # print(loss)

    # 使用第一个图数据的测试掩码和标签进行评估
    _, predicted = torch.max(out[test_mask], 1)
    y_true = mixed_data[0].y[test_mask].to(device)
    # 计算指标
    accuracy = accuracy_score(y_true.cpu(), predicted.cpu())
    precision = precision_score(y_true.cpu(), predicted.cpu(), average='weighted', zero_division=1)
    recall = recall_score(y_true.cpu(), predicted.cpu(), average='weighted', zero_division=1)

    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss,accuracy,precision,recall

class traditional_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(traditional_GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.local_attention = nn.MultiheadAttention(embed_dim=38, num_heads=1)

    def forward(self, mixed_data):
        data = mixed_data[0]
        x, edge_index = data.x, data.edge_index
        # x,_=self.local_attention(x,x,x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # x.squeeze() #F.log_softmax(x, dim=1)

# Define our model
class FusionGAT(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_classes):
        super(FusionGAT, self).__init__()
        # Initialize a shared GAT layer
        # 初始化一个共享的图注意力网络层
        self.gat1 = GATConv(num_node_features, hidden_dim)

        # Define two GATConv layers for processing different inputs
        # 定义两个GATConv层用于处理不同的输入
        self.gat2_x = GATConv(hidden_dim, output_dim, heads=1)
        self.gat2_y = GATConv(hidden_dim, output_dim, heads=1)

        # Fully connected layer for dimension adjustment and classification
        # 全连接层，用于维度调整和分类
        self.fc = nn.Linear(2 * output_dim + num_node_features, num_classes)

        # An additional fully connected layer for dimension adjustment (optional)
        # 一个额外的全连接层用于维度调整（可选）
        # self.adjust_dim = nn.Linear(num_node_features, output_dim)

        # Define self-attention layers for local feature processing
        # 定义自注意力层，用于局部特征处理
        self.local_attention1 = nn.MultiheadAttention(embed_dim=num_node_features, num_heads=1)
        self.local_attention2 = nn.MultiheadAttention(embed_dim=num_node_features, num_heads=1)

        # Global self-attention layer for feature fusion (optional)
        # 全局自注意力层，用于特征融合（可选）
        self.global_attention = nn.MultiheadAttention(embed_dim=2 * output_dim, num_heads=1)

    def forward(self, mixed_data):
        data_x, data_y = mixed_data[0], mixed_data[1]

        # Process the first input axis
        # 处理第一个输入轴
        x, edge_index_x = data_x.x, data_x.edge_index
        x_res = x  # Save original features for residual connections
        # 保存原始特征用于残差连接

        # Self-attention for x
        # x的自注意力模块
        x, _ = self.local_attention1(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x.squeeze(0)
        x = F.relu(self.gat1(x, edge_index_x))
        x = F.dropout(x, training=self.training)
        x = self.gat2_x(x, edge_index_x)

        # Process the second input axis
        # 处理第二个输入轴
        y, edge_index_y = data_y.x, data_y.edge_index

        # Self-attention for y
        # y的自注意力模块
        y, _ = self.local_attention2(y.unsqueeze(0), y.unsqueeze(0), y.unsqueeze(0))
        y = y.squeeze(0)
        y = F.relu(self.gat1(y, edge_index_y))
        y = F.dropout(y, training=self.training)
        y = self.gat2_y(y, edge_index_y)

        fusion = torch.cat((x, y), dim=1)
        # Fusion of features from two axes with optional global self-attention
        # 两个轴的特征融合，可选使用全局自注意力
        # fusion=self.global_attention(fusion.unsqueeze(0),fusion.unsqueeze(0),fusion.unsqueeze(0))
        fusion = torch.cat([fusion, x_res], dim=1)

        # Classification
        # 分类
        out = self.fc(fusion)
        return F.log_softmax(out, dim=1)

if __name__ == '__main__':
    # 构建虚拟图数据
    num_nodes = 5075
    class_n = 3
    split_ratio = 0.8

    # 生成虚拟数据并进行拆分以进行对比实验
    data1, data2 = generate_dummy_graph_data(num_nodes=100, num_node_features=19, num_classes=3, split_ratio=0.8)
    X = data1.x.numpy()
    y = data1.y.numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Comparison Experiment
    results = compare_method(X_train, X_test, y_train, y_test)

    # 初始化模型、优化器和损失函数
    device = torch.device('cpu')
    model = FusionGAT(num_node_features=19, hidden_dim=64, output_dim=16, num_classes=class_n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    mixed_data = [data1.to(device), data2.to(device)]

    epochs = 200  # 或者根据需要调整
    for epoch in range(1, epochs + 1):
        loss,accuracy,precision,recall = train(mixed_data, model, optimizer, criterion)
        # val_acc = val(mixed_data, model, criterion)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    # Save the model
    model_save_path = 'model_fusion.pth'  # Define your path and file name
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')




