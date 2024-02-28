import os
from typing import List, Tuple

from llama_index.core import text_splitter, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import TextNode

# %%
# =============================================================================
# Get Nodes
# =============================================================================
def get_nodes(docs: List[Document])-> Tuple[TextNode]:
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text", )
    # extract both the sentence window nodes and the base nodes
    nodes = node_parser.get_nodes_from_documents(docs)
    base_nodes = text_splitter.get_nodes_from_documents(docs)
    return nodes, base_nodes