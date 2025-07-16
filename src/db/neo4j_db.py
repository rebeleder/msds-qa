import numpy as np
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from py2neo import Graph, Node, Relationship

from src.config import hp


class Neo4jDB:

    def __init__(
        self,
        embed_model: Embeddings,
        bolt_url: str = hp.neo4j_bolt_url,
        username: str = hp.neo4j_username,
        password: str = hp.neo4j_password,
    ) -> None:
        self.embed_model: Embeddings = embed_model
        self.bolt_url: str = bolt_url
        self.username: str = username
        self.password: str = password

        self.graph: Graph = self.get_graph()

    def get_graph(self) -> Graph:
        """获取Neo4j数据库连接"""
        return Graph(self.bolt_url, auth=(self.username, self.password))

    def create_node(
        self,
        label: str,
        name: str,
        content: str,
        context: str,
        **properties: dict[str, object],
    ) -> Node:
        """
        创建或合并一个节点.

        :param label: 节点类别
        :param content: 节点内容
        :param context: 节点上下文
        :param properties: 节点属性
        :return: 创建或合并的节点
        """
        embed: list[float] = self.get_node_embedding(content)

        node = Node(label, name=name, context=context, embed=embed, **properties)

        self.graph.merge(node, label, "name")
        return node

    def create_edge(
        self,
        start_node_name: str,
        end_node_name: str,
        rel_type: str,
        description: str = "",
    ) -> Relationship:
        """
        在两个节点之间创建一个边
        """
        embed = self.get_edge_embedding(rel_type)

        start_node = self.get_node_by_name(start_node_name)
        end_node = self.get_node_by_name(end_node_name)

        try:
            # ! 判断节点是否存在
            rel = Relationship(
                start_node,
                rel_type,
                end_node,
                embed=embed,
                description=description,
            )
            self.graph.merge(rel)
        except Exception as e:
            print(f"Error creating edge: {e}")

    def get_node_by_name(self, name: str) -> Node | None:
        """根据节点名称获取节点"""

        return self.graph.nodes.match(name=name).first()

    def get_node_embedding(self, text: str) -> list[float]:
        """
        获取节点的嵌入向量
        """
        return self.embed_model.embed_query(text)

    def get_nodes_embedding(self, nodes: list[Node]) -> np.ndarray:
        embeds = [node["embed"] for node in nodes]
        return np.array(embeds)

    def get_edges_embedding(self, edges: list[Relationship]) -> np.ndarray:
        embeds = [rel["embed"] for rel in edges]
        return np.array(embeds)

    def get_edge_embedding(self, text: str) -> list[float]:
        """
        获取边的嵌入向量
        """
        return self.embed_model.embed_query(text)

    def delete_all(self) -> None:
        """
        删除整个图数据库
        """
        self.graph.delete_all()

    # ! 当前阶段仅支持单文本,少量向量检索
    def get_relevant_chunks(
        self, query: str | list[str], limit: int = 10
    ) -> list[Document]:
        query = query if isinstance(query, list) else [query]
        query_embedding = self.embed_model.embed_documents(query)
        nodes = list(self.graph.nodes.match())
        embeds = self.get_nodes_embedding(nodes)

        cosine = query_embedding @ embeds.T
        top_indices = np.argsort(cosine, axis=1)[:, -limit:][:, ::-1]
        top_nodes = np.array(nodes)[top_indices]

        top_nodes = top_nodes.flatten().tolist()[:limit]

        # chunks = [
        #     Document(page_content=rel["description"])
        #     for node in top_nodes
        #     for rel in self.graph.relationships.match((node, None))
        # ]
        chunks = []
        for node in top_nodes:
            if node["context"] is None:
                continue
            chunks.append(Document(page_content=node["context"]))

            rels = list(self.graph.relationships.match(nodes=(node, None)))
            for rel in rels:
                _start_node = rel.start_node
                _end_node = rel.end_node

                chunks.append(
                    Document(
                        page_content=f"关系: {_start_node['name']} -> {rel.__class__.__name__}, 描述: {_end_node['context']}"
                    )
                )

        return chunks

if __name__ == "__main__":
    from src.model import OllamaClient

    embed_mode = OllamaClient().get_embed_model()
    db = Neo4jDB(embed_mode)
    node1 = db.create_node(
        label="PERSON", name="张三", content="张三", context="张三是一个程序员"
    )
    node2 = db.create_node(
        label="PERSON", name="李四", content="李四", context="李四是一个设计师"
    )
    edge = db.create_edge("张三", "李四", "同学", "张三和李四是同学关系")

    node3 = db.create_node(
        label="PERSON", name="王五", content="王五", context="王五是一个产品经理"
    )
    edge2 = db.create_edge("张三", "王五", "朋友", "张三和王五是朋友关系")

    node4 = db.create_node(
        label="POSITION", name="北京", content="北京", context="北京是中国的首都"
    )
    edge3 = db.create_edge("张三", "北京", "居住地", "张三居住在北京")

    print(db.get_relevant_chunks("张三"))
