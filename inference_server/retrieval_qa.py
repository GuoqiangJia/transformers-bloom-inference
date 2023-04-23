import glob
from abc import ABC, abstractmethod

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.redis import Redis
from .constants import redis_url


class EmbeddingDir(ABC):

    def __init__(self, directory: str) -> None:
        super().__init__()
        self.directory = directory

    @abstractmethod
    def embedding(self):
        """embedding all files under the directory"""


class RedisEmbedding(EmbeddingDir):

    def __init__(self, directory: str, spacy_pipeline='en_core_web_trf') -> None:
        super().__init__(directory)
        try:
            import spacy
            spacy.prefer_gpu()
        except ImportError:
            raise ImportError("Spacy is not installed, please install it with `pip install spacy`.")

        self.text_splitter = SpacyTextSplitter(pipeline=spacy_pipeline)
        self.huggingEmbedding = HuggingFaceInstructEmbeddings()
        self.redis_url = constants.redis_url

    def embedding(self):
        all_files = glob.glob(self.directory + "/*.csv")
        for filename in all_files:
            with open(filename, 'r', encoding='utf-8') as f:
                texts = self.text_splitter.split_text(f.read())
                docs = [Document(page_content=t, metadatas={"source": f"{filename}-{i}-pl"}) for i, t in
                        enumerate(texts)]
        search_index = Redis(redis_url=self.redis_url).add_documents(docs, embeddings=self.huggingEmbedding)
        print(search_index)
        return search_index


if __name__ == '__main__':
    em = RedisEmbedding('../assets/corpus/audio_txt_clean')
    em.embedding()
