import os
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import chromadb
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Llama Index imports
from llama_index.core import (Document, StorageContext,
                              VectorStoreIndex, Settings, SimpleDirectoryReader)
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SentenceSplitter,
    TokenTextSplitter)
from llama_index.core.schema import BaseNode
from llama_index.readers.file import CSVReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index_client import ChromaVectorStore
from llama_parse import LlamaParse
from llama_index.core.extractors import (TitleExtractor,
                                         QuestionsAnsweredExtractor,
                                         SummaryExtractor, KeywordExtractor)
from llama_index.extractors.entity import EntityExtractor

# =============================================================================
# create
# %%
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-3-small"
llm = OpenAI(temperature=0.1, model=model_name, max_tokens=512)
embed_model = OpenAIEmbedding(model=embedding_model_name)
Settings.llm = llm
Settings.embed_model = embed_model

# base node parser is a sentence splitter
text_splitter = SentenceSplitter()
Settings.text_splitter = text_splitter


# %%
# =============================================================================
# %%
# =============================================================================
# Load data
def data_loader(filepath: str, parse: bool = False) -> List[Document]:
    """
    Loads and parses a PDF file using LlamaParse.

    This function initializes a LlamaParse parser with specified settings, including
    the result type as 'markdown' and an API key obtained from the environment variables.
    It then loads and parses the PDF file located at the given filepath using the parser,
    returning the parsed document(s).

    Parameters:
    - filepath (str): The path to the PDF file to be loaded and parsed.

    Returns:
    - str: The parsed document(s) in the specified result type.
    """

    _, file_extension = os.path.splitext(filepath)
    if file_extension == ".pdf" and parse:
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_PARSER_API_KEY"),
            result_type="markdown",
            verbose=True, )

        file_extractor = {".pdf": parser}
        docs = SimpleDirectoryReader(
            filepath, file_extractor=file_extractor).load_data()
    else:
        docs = SimpleDirectoryReader(filepath).load_data()

    return docs


def csv_excel_data_loader(filepath: Path, embed_cols: Optional[str] = None,
                          embed_metadata: Optional[str] = None) -> Tuple[
                            List[Document], Document]:
    """
    Reads .csv and .xlsx data from a file and processes columns for embedding and metadata extraction.

    This function reads data from a specified file (CSV or Excel) and processes columns
    for embedding and metadata extraction. It creates Document objects based on the specified
    columns and metadata, combining them into a single Document object if embedding is enabled.

    Parameters:
    - filepath (Path): The path to the file to read data from.
    - embed_cols (Optional[str]): Columns to process for embedding.
    - embed_metadata (Optional[str]): Columns to extract metadata from.

    Returns:
    - Tuple[List[Document], Document]: A tuple containing a list of Document objects created
      based on the specified columns and metadata, and a single Document object combining
      text from all documents if embedding is enabled.
    """
    docs = []
    document = None
    df = None
    _, file_extension = os.path.splitext(filepath)

    if file_extension == ".csv":
        df = pd.read_csv(filepath)
    elif file_extension == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        raise ValueError("File must be a CSV or Excel file")

    if embed_cols:
        for _, row in df.iterrows():
            to_metadata = {col: row[col] for col in embed_metadata if col in row}
            values_to_embed = {k: str(row[k]) for k in embed_cols if k in row}
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            newdoc = Document(text=to_embed, metadata=to_metadata)
            docs.append(newdoc)

        document = Document(text="\n\n".join([doc.text for doc in docs]))

    elif file_extension == ".csv":
        docs = CSVReader(concat_rows=False).load_data(file=filepath)

    return docs, document


def load_data_to_sql_db(filepath: str, dbpath: str, tablename: str) -> None:
    """
    Loads data from a CSV file into an SQL database table.

    This function reads data from a specified CSV file into a pandas DataFrame. It then
    establishes a connection to an SQLite database located at the given database path.
    Using this connection, the function writes the contents of the DataFrame to a specified
    table within the SQLite database. If the table already exists, its contents are replaced
    with the data from the DataFrame.

    Parameters:
    - filepath (str): The path to the CSV/xlsx file containing the data to be loaded.
    - dbpath (str): The path to the SQLite database file where the data will be stored.
    - tablename (str): The name of the table within the SQLite database where the data
      from the CSV file will be written.

    Returns:
    - None
    """
    _, file_extension = os.path.splitext(filepath)
    if file_extension == ".csv":
        # Read the data from the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
    elif file_extension == ".xlsx":
        # Read the data from the Excel file into a pandas DataFrame
        df = pd.read_excel(filepath)

    else:
        raise ValueError("File must be a CSV or Excel file")

    # Create a connection to the SQLite database and an engine
    conn = sqlite3.connect(dbpath)
    engine = create_engine("sqlite:///" + dbpath)

    # Write the DataFrame to a table in the SQLite database
    df.to_sql(tablename, engine, if_exists='replace', index=False)

    # It's good practice to close the connection when done
    conn.close()
    return None


# =============================================================================
# %%
# =============================================================================
# Get Nodes
def get_nodes(docs: List[Document]) -> Tuple[List[BaseNode], List[BaseNode]]:
    """
    Extracts nodes from documents using both a SentenceWindowNodeParser and a base text splitter.

    This function initializes a SentenceWindowNodeParser with default settings, specifically
    a window size of 3. It then uses this parser along with a base text splitter to extract
    nodes from the provided documents. The SentenceWindowNodeParser extracts nodes based on
    sentence windows, while the base text splitter extracts base nodes without considering
    sentence windows.

    Parameters:
    - docs (List[Document]): A list of Document objects from which nodes are to be extracted.

    Returns:
    - Tuple[List[TextNode], List[TextNode]]: A tuple containing two lists of TextNode objects.
      The first list contains nodes extracted by the SentenceWindowNodeParser, and the second
      list contains base nodes extracted by the base text splitter.
    """
    # Initialize the SentenceWindowNodeParser with default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,  # The size of the sentence window
        window_metadata_key="window",  # Metadata key for window information
        original_text_metadata_key="original_text",  # Metadata key for original text information
    )

    # Extract nodes using the SentenceWindowNodeParser
    nodes = node_parser.get_nodes_from_documents(docs)

    # Extract base nodes using the base text splitter
    base_nodes = SentenceSplitter.get_nodes_from_documents(docs)

    return nodes, base_nodes


# =============================================================================
# %%
# Get Index
def get_index(vector_db_path, collection_name, nodes=None):
    db = chromadb.PersistentClient(path=vector_db_path)
    textsplitter = TokenTextSplitter(separator=" ",
                                     chunk_size=512, chunk_overlap=128)
    title_extractor = TitleExtractor(nodes=5, llm=llm)
    # qa_extractor = QuestionsAnsweredExtractor(questions=3)
    summary_extractor = SummaryExtractor(summaries=["prev", "self"], llm=llm)
    keyword_extractor = KeywordExtractor(keywords=10, llm=llm),
    entity_extractor = EntityExtractor(
        prediction_threshold=0.5,
        label_entities=False,  # include the entity label in the metadata (can be erroneous)
        device="cpu",  # set to "cuda" if you have a GPU
    )
    # Check if the collection does not exist
    if collection_name not in [col.name for col in db.list_collections()]:
        print("building index", collection_name)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context,
                                 embed_model=embed_model,
                                 transformations=[textsplitter,
                                                  summary_extractor,
                                                  title_extractor,
                                                  entity_extractor],
                                 show_progress=True)
    else:
        # This block now correctly handles the case where the
        # collection already exists
        print("loading index", collection_name)
        chroma_collection = db.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model,
                                                   transformations=[textsplitter,
                                                                    summary_extractor,
                                                                    title_extractor,
                                                                    entity_extractor,
                                                                    keyword_extractor],
                                                   show_progress=True)

    return index


# =============================================================================
# get query engine
def get_sentence_window_query_engine(
        index,
        similarity_top_k=6,
        rerank_top_n=2, ):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base")

    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank])
    return query_engine
# =============================================================================
