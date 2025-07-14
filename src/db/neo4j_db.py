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

        self.embed_model: Embeddings = embed_model
        self.graph = self.get_db()

    def get_db(self) -> Graph:
        """获取Neo4j数据库连接"""
        return Graph(self.bolt_url, auth=(self.username, self.password))

    def create_node(self, label, context, **properties) -> Node:
        """
        创建或合并一个节点.

        :param label: 节点标签
        :param context: 节点上下文
        :param properties: 节点属性
        :return: 创建或合并的节点
        """
        embed = self.get_node_embedding(label)
        # TODO
        node = Node(
            label,
            embed=embed,
            context=context,
            **properties,
        )
        self.graph.merge(node, label, "name")
        return node

    def create_edge(
        self, start_node: Node, end_node: Node, rel_type: str
    ) -> Relationship:
        """
        在两个节点之间创建一个边
        """
        embed = self.get_edge_embedding(rel_type)
        rel = Relationship(start_node, rel_type, end_node, embed=embed)
        self.graph.merge(rel)
        return rel

    # TODO合并好还是不合并好
    def get_node_embedding(self, text) -> list[float]:
        """
        获取节点的嵌入向量
        """
        return self.embed_model.embed_query(text)

    def get_edge_embedding(self, text) -> list[float]:
        """
        获取边的嵌入向量
        """
        return self.embed_model.embed_query(text)

    def delete_all(self) -> None:
        """
        删除整个图数据库
        """
        self.graph.delete_all()


if __name__ == "__main__":
    from src.model import OllamaClient

    embed_mode = OllamaClient().get_embed_model()
    db = Neo4jDB(embed_mode)
    node1 = db.create_node("人", name="张三")
    node2 = db.create_node("人", name="李四")
    edge = db.create_edge(node1, node2, "同学")

    node3 = db.create_node("人", name="王五")
    edge2 = db.create_edge(node1, node3, "朋友")

    node4 = db.create_node("地点", name="北京")
    edge3 = db.create_edge(node1, node4, "居住地")
    print("Neo4j数据库已初始化")
