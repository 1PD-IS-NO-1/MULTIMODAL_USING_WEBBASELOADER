import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import re

# API Keys
PINECONE_API_KEY = "PASTE YOUR PINECOND API KEY"
GOOGLE_API_KEY = "PASTE YOUR GEMINI API KEY"

# Document sources
doc_sources = {
    "langsmith": "https://docs.smith.langchain.com/",
    "langchain": "https://python.langchain.com/docs/get_started/introduction",
    "llamaindex": "https://docs.llamaindex.ai/en/stable/",
    "crewai": "https://docs.crewai.com/",
    "sentence-transformers": "https://www.sbert.net/docs/quickstart.html"
}

class DocumentationManager:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = "test1"
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GOOGLE_API_KEY
        )
        self.ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        self.vectorstore = None
        self.initialize_vectorstore()

    def ensure_index_exists(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

    def initialize_vectorstore(self):
        self.vectorstore = PineconeVectorStore(
            self.index,
            self.embeddings,
            "text"
        )

    def update_documentation(self):
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for source_name, url in doc_sources.items():
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)
                for doc in split_docs:
                    doc.metadata['source'] = source_name
                documents.extend(split_docs)
            except Exception as e:
                print(f"Error loading {source_name}: {e}")

        # Delete existing vectors and add new ones
        self.index.delete(deleteAll=True)
        self.vectorstore.add_documents(documents)
        return len(documents)

    def get_retriever_tools(self):
        retriever_tools = []
        for source_name in doc_sources.keys():
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "filter": {"source": source_name},
                    "k": 3
                }
            )
            tool = create_retriever_tool(
                retriever,
                f"{source_name}_search",
                f"Search for information about {source_name}. Use this tool for {source_name}-specific questions!"
            )
            retriever_tools.append(tool)
        return retriever_tools

class AgentManager:
    def __init__(self, doc_manager: DocumentationManager):
        self.doc_manager = doc_manager
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        self.agent_executor = self.initialize_agent()

    def initialize_agent(self):
        # Set up Wikipedia tool
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        wiki_tool = Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Useful for querying Wikipedia to get additional information on various topics."
        )

        # Combine all tools
        tools = [wiki_tool] + self.doc_manager.get_retriever_tools()

        # Set up the prompt template
        prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        {agent_scratchpad}""")

        # Create the agent and executor
        agent = create_react_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize managers
doc_manager = DocumentationManager()
agent_manager = AgentManager(doc_manager)
