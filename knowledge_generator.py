import os
import re
import json
import numpy as np
import hashlib
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import pickle
import faiss

class LocalKnowledgeBase:
    """完全本地化的知识库系统，无需外部数据库"""
    def __init__(self, model,cue,storage_dir: str = "knowledge_db"):
        """
        初始化本地知识库
        
        参数:
            storage_dir: 本地存储目录
        """
        # 创建存储目录
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        
        # 1. 初始化嵌入模型 (中文优化)
        self.embedding_model = SentenceTransformer('moka-ai/m3e-base')
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # 2. 初始化 FAISS 向量索引
        self.index_file = os.path.join(storage_dir, "knowledge_index.faiss")
        self.metadata_file = os.path.join(storage_dir, "knowledge_metadata.pkl")
        
        # 尝试加载现有索引
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"加载现有知识库，包含 {len(self.metadata)} 个知识片段")
        else:
            # 创建新索引
            self.index = faiss.IndexFlatIP(self.vector_size)  # 内积相似度
            self.metadata = []
            print("创建新知识库")
        
        # 3. 初始化 CAMEL AI 模型
        self.model = model
        
        # 4. 创建聊天代理
        self.agent = ChatAgent(
            model=self.model,
            system_message = cue
        )
    
    def _chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 20) -> List[str]:
        """
        将文本分割为带重叠的块
        
        参数:
            text: 输入文本
            chunk_size: 每个块的最大长度
            overlap: 块之间的重叠字符数
            
        返回:
            文本块列表
        """
        # 使用句子分割更智能的分块
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果添加当前句子不会超过块大小
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                # 保存当前块并开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
                # 如果句子本身超过块大小，强制分割
                if len(current_chunk) > chunk_size:
                    words = current_chunk.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= chunk_size:
                            current_chunk += word + " "
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = word + " "
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _save_to_disk(self):
        """保存索引和元数据到磁盘"""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"知识库已保存到 {self.storage_dir}")
    
    def add_document(self, file_path: str):
        """
        添加文档到知识库
        
        参数:
            file_path: 文本文件路径
        """
        # 1. 读取文件内容
        with open(file_path, "r") as f:
            text = f.read()
        
        # 2. 清理文本（移除多余空格和换行）
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 3. 分割文本
        chunks = self._chunk_text(text)
        
        # 4. 生成嵌入向量
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        
        # 5. 添加到索引和元数据
        start_idx = len(self.metadata)
        
        # 为每个块创建元数据
        for idx, chunk in enumerate(chunks):
            self.metadata.append({
                "text": chunk,
                "source": os.path.basename(file_path),
                "chunk_id": start_idx + idx
            })
        
        # 添加到 FAISS 索引
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings)
        
        print(f"已添加文档: {file_path} ({len(chunks)} 个块)")
        
        # 6. 保存到磁盘
        self._save_to_disk()
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索与查询相关的上下文
        
        参数:
            query: 查询文本
            top_k: 返回的结果数量
            
        返回:
            相关上下文列表
        """
        # 1. 生成查询嵌入
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        
        # 确保二维数组
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 2. 执行相似度搜索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 3. 格式化结果
        context = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS 返回 -1 表示无效索引
                metadata = self.metadata[idx]
                context.append({
                    "text": metadata["text"],
                    "source": metadata["source"],
                    "score": float(1 - distances[0][i])  # 转换为相似度分数
                })
        
        # 按分数排序
        context.sort(key=lambda x: x["score"], reverse=True)
        return context
    
    def generate_response(self, query: str) -> str:
        """
        生成知识库回答
        
        参数:
            query: 用户查询
            
        返回:
            AI 生成的回答
        """
        # 1. 检索相关上下文
        context = self.retrieve_context(query)
        
        # 2. 构建上下文提示
        context_str = "\n\n".join([f"[来源: {c['source']}]\n{c['text']}" for c in context])
        prompt = f"""
        基于以下上下文信息回答问题。如果上下文不包含答案，请说明你不知道。

        上下文：
        {context_str}

        问题：{query}
        """
        
        # 3. 获取 AI 回答
        response = self.agent.step(prompt)
        return response.msg.content
    
    def query_knowledge(self, query: str, show_context: bool = False) -> str:
        """
        查询知识库并返回完整结果
        
        参数:
            query: 用户查询
            show_context: 是否显示上下文来源
            
        返回:
            格式化后的回答
        """
        # 1. 获取回答
        response = self.generate_response(query)
        
        # 2. 获取上下文
        context = self.retrieve_context(query)
        
        # 3. 格式化输出
        result = f"问题: {query}\n\n回答: {response}"
        
        if show_context:
            context_info = "\n".join(
                [f"- 来源: {c['source']} (相关性: {c['score']:.2f})\n  {c['text'][:100]}..." 
                 for c in context]
            )
            result += f"\n\n上下文来源:\n{context_info}"
        
        return result
    
    def export_to_json(self, output_file: str = "knowledge_export.json"):
        """导出知识库到 JSON 文件"""
        export_data = {
            "metadata": self.metadata,
            "storage_dir": self.storage_dir,
            "vector_size": self.vector_size
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"知识库已导出到 {output_file}")
    
    def import_from_json(self, input_file: str):
        """从 JSON 文件导入知识库"""
        with open(input_file, "r", encoding="utf-8") as f:
            import_data = json.load(f)
        
        # 重建索引
        self.vector_size = import_data["vector_size"]
        self.index = faiss.IndexFlatIP(self.vector_size)
        self.metadata = import_data["metadata"]
        
        # 重新生成嵌入
        all_texts = [item["text"] for item in self.metadata]
        embeddings = self.embedding_model.encode(all_texts, convert_to_tensor=False)
        self.index.add(embeddings)
        
        # 保存到磁盘
        self._save_to_disk()
        print(f"知识库已从 {input_file} 导入")

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='./.env')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    url=os.getenv('SILICONFLOW_BASE_URL'),
    api_key=os.getenv('SILICONFLOW_API_KEY'),
    model_config_dict={
        # 核心参数
        "temperature": 0.1,        # 控制随机性 (0-1)
        "max_tokens": 512,         # 最大生成长度
    }
)
# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 1. 初始化本地知识库
    knowledge_base = LocalKnowledgeBase(model,"你是一个知识库助手，根据提供的上下文回答问题")
    
    # 2. 添加文档到知识库
    knowledge_base.add_document("all_solution.txt")
    # 可以添加更多文档：knowledge_base.add_document("another_doc.txt")
    
    # 3. 导出知识库（可选）
    knowledge_base.export_to_json()
    
    # 4. 查询知识库
    while True:
        print("\n" + "="*50)
        query = input("请输入问题 (输入 'exit' 退出): ")
        
        if query.lower() in ["exit", "quit"]:
            break
            
        response = knowledge_base.query_knowledge(query, show_context=True)
        print("\n" + response)