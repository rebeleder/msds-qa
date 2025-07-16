import asyncio
import html
import itertools
import re
from collections import defaultdict

from langchain.schema import Document
from langchain_core.embeddings import Embeddings

from src.db import Neo4jDB
from src.memory import ChatMessages
from src.model import OllamaClient, SiliconflowClient
from src.parser import MsdsParser
from src.prompt import Prompt


def is_float_regex(value: str):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


client = SiliconflowClient()


chat_model = client.get_chat_model()
embed_model = client.get_embed_model()


def clean_str(input: str) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


async def _handle_single_entity_extraction(record_attributes: list[str]):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
    )


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


async def _handle_single_relationship_extraction(record_attributes: list[str]):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
    )


class Msds2GraphDB:

    def __init__(
        self,
        files: list[str] | str,
        db: Neo4jDB = Neo4jDB(embed_model),
    ) -> None:
        self.files: list[str] = files if isinstance(files, list) else [files]
        self.parser = MsdsParser
        self.documents = self.get_documents()
        self.db: Neo4jDB = db
        self.embed_model: Embeddings = self.db.embed_model

        self.continue_prompt = self._get_continue_prompt()
        self.if_loop_prompt = self._get_if_loop_prompt()

    def get_documents(self) -> list[Document]:
        documents = self.parser(self.files).invoke()
        return documents

    def _get_hint_prompt(self, input_text: str) -> str:
        prompt = Prompt.get_prompt("entity_extraction").format(
            tuple_delimiter=Prompt.get_default_tuple_delimiter(),
            record_delimiter=Prompt.get_default_record_delimiter(),
            completion_delimiter=Prompt.get_default_completion_delimiter(),
            entity_types=Prompt.get_default_entity_types(),
            input_text=input_text,
        )
        return prompt

    def _get_continue_prompt(self) -> str:
        prompt = Prompt.get_prompt("entity_continue_extraction")
        return prompt

    def _get_if_loop_prompt(self) -> str:
        prompt = Prompt.get_prompt("entity_if_loop_extraction")
        return prompt

    async def parse_single_document(self, context: str):
        memory_view = ChatMessages()

        hint_prompt = self._get_hint_prompt(context)
        memory_view.add_user_message(hint_prompt)

        result = await chat_model.ainvoke(memory_view.get_messages())
        memory_view.add_ai_message(result)

        for idx in range(hp.max_retry):
            memory_view.add_user_message(self.continue_prompt)

            glean_result = await chat_model.ainvoke(memory_view.get_messages())
            memory_view.add_ai_message(glean_result)

            if idx == hp.max_retry - 1:
                break

            memory_view.add_user_message(self.if_loop_prompt)

            if_loop_result = await chat_model.ainvoke(memory_view)

            if_loop_result: bool = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            "".join(memory_view.get_ai_messages()),
            [
                Prompt.get_default_record_delimiter(),
                Prompt.get_default_completion_delimiter(),
            ],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [Prompt.get_default_tuple_delimiter()]
            )
            if_entities = await _handle_single_entity_extraction(record_attributes)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            if_relation = await _handle_single_relationship_extraction(
                record_attributes
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        return dict(maybe_nodes), dict(maybe_edges)

    async def __call__(self, chunks: list[str] | str):
        chunks = chunks if isinstance(chunks, list) else [chunks]

        results = await asyncio.gather(*[self.parse_single_document(c) for c in chunks])
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)

        maybe_nodes = list(itertools.chain.from_iterable(maybe_nodes.values()))
        maybe_edges = list(itertools.chain.from_iterable(maybe_edges.values()))

        for node in maybe_nodes:
            self.db.create_node(
                category=node["entity_type"],
                content=node["entity_name"],
                context=node["description"],  # TODO需要支持context
                description=node["entity_name"],
            )

        for edge in maybe_edges:
            self.db.create_edge(
                start_node_name=edge["src_id"],
                end_node_name=edge["tgt_id"],
                rel_type=edge["keywords"],
            )
        return maybe_nodes, maybe_edges


async def main():
    from src.config import hp
    from src.toolkits import get_files_from_kb_space

    files = get_files_from_kb_space(hp.knowledge_file_path)[:5]

    pipe = Msds2GraphDB(files)
    await pipe(
        "小王是一个程序员，他的工作是编写代码。他喜欢吃苹果和香蕉, 小李是一个设计师，他喜欢画画和摄影, 小张是一个医生，他喜欢看书和旅游。小王和小李是好朋友，他们经常一起出去玩。小张和小王是同事，他们在同一家公司工作。"
    )


if __name__ == "__main__":
    import asyncio

    from src.config import hp
    from src.toolkits import get_files_from_kb_space

    files = get_files_from_kb_space(hp.knowledge_file_path)[:5]

    asyncio.run(main())
