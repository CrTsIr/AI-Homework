import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.types import ModelType
from typing import List, Dict, Any

# ======================
# Matplotlib 工具函数集
# ======================


def plot_sequence(data: List[float], title: str = "Sequence Plot", xlabel: str = "Index", ylabel: str = "Value") -> str:
    """
    Plot a numerical sequence. Returns the file path of the saved image.
    
    Args:
        data: List of numerical values to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    try:
        # 创建图形
        plt.figure(figsize=(10, 6))
        plt.plot(data, 'o-', markersize=8, linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图像
        file_path = "plot.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
        return f"Sequence plot saved to: {file_path}"
    
    except Exception as e:
        return f"Failed to plot sequence: {str(e)}"

def plot_tree(nodes: List[Dict], edges: List[Dict], title: str = "Tree Structure") -> str:
    """
    Plot a tree structure. Returns the file path of the saved image.
    
    Args:
        nodes: List of node dicts with 'id' and optional 'label'
        edges: List of edge dicts with 'parent' and 'child'
        title: Plot title
    """
    try:
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 创建树结构
        G = nx.DiGraph()
        pos = {}
        levels = {}
        
        # 添加节点并计算层级
        for node in nodes:
            node_id = node['id']
            G.add_node(node_id, label=node.get('label', node_id))
            
            # 找到根节点（没有父节点的节点）
            if not any(edge['child'] == node_id for edge in edges):
                levels[node_id] = 0
        
        # 计算节点位置
        def assign_positions(node_id, x, y):
            pos[node_id] = (x, y)
            children = [edge['child'] for edge in edges if edge['parent'] == node_id]
            
            if children:
                # 计算水平间距
                total_width = len(children) * 1.0
                start_x = x - total_width / 2
                
                for i, child in enumerate(children):
                    child_x = start_x + i
                    assign_positions(child, child_x, y - 1)
        
        # 从根节点开始分配位置
        roots = [node for node in G.nodes if levels.get(node, -1) == 0]
        for i, root in enumerate(roots):
            assign_positions(root, i * 2, 0)
        
        # 添加边
        for edge in edges:
            G.add_edge(edge['parent'], edge['child'])
        
        # 绘制树
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, 
                node_color="skyblue", font_size=10, font_weight="bold", 
                arrowsize=20, arrowstyle='->', width=2)
        
        plt.title(title, fontsize=14)
        
        # 保存图像
        file_path = "plot.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
        return f"Tree plot saved to: {file_path}"
    
    except Exception as e:
        return f"Failed to plot tree: {str(e)}"

def plot_graph(nodes: List[Dict], edges: List[Dict], title: str = "Graph Structure", 
               directed: bool = False) -> str:
    """
    Plot a graph structure. Returns the file path of the saved image.
    
    Args:
        nodes: List of node dicts with 'id' and optional 'label'
        edges: List of edge dicts with 'from' and 'to'
        title: Plot title
        directed: Whether the graph is directed
    """
    try:
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 创建图结构
        G = nx.DiGraph() if directed else nx.Graph()
        
        # 添加节点
        for node in nodes:
            node_id = node['id']
            G.add_node(node_id, label=node.get('label', node_id))
        
        # 添加边
        for edge in edges:
            G.add_edge(edge['from'], edge['to'], label=edge.get('label', ''))
        
        # 计算布局
        pos = nx.spring_layout(G, seed=42)  # 使用固定种子保证可重复性
        
        # 绘制图
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightgreen")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")
        
        if directed:
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, 
                                  width=1.5, edge_color="gray")
        else:
            nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray")
        
        # 添加边标签
        edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
        
        plt.title(title, fontsize=14)
        plt.axis('off')
        
        # 保存图像
        file_path = "plot.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
        return f"Graph plot saved to: {file_path}"
    
    except Exception as e:
        return f"Failed to plot graph: {str(e)}"

# ======================
# Agent 创建与配置
# ======================
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='./.env')

from camel.models import ModelFactory
from camel.types import ModelPlatformType
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    url=os.getenv('SILICONFLOW_BASE_URL'),
    api_key=os.getenv('SILICONFLOW_API_KEY'),
    model_config_dict={
        # 核心参数
        "temperature": 0.9,        # 控制随机性 (0-1)
        "max_tokens": 2048,         # 最大生成长度
    }
)

print("请输入题面的文件名")
statement_file=input()

with open(statement_file, 'r',encoding='utf-8', errors='ignore') as file:
    statement_text = file.read()


# 创建数学可视化 Agent
visualization_agent = ChatAgent(
    system_message=(
        "你是一个绘图助手，阅读题面{"+statement_text+"}，并根据用户提供的要求，选用合适的工具进行绘图"
        "如果你无法理解用户的问题，可以如实报告并询问一些更详细的信息。"
        "如果用户的要求不恰当，你可以回绝。"
    ),
    model=model,
    tools={plot_sequence,plot_tree,plot_graph}
)

while True:
    print("\n" + "="*50)
    query = input("对话中 (输入 'exit' 退出): ")
    
    if query.lower() in ["exit", "quit"]:
        break

    response = visualization_agent.step(query)
    print(response.msg.content)
