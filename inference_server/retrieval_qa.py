import glob
import logging
from abc import ABC, abstractmethod

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from redis_fix import Redis
from constants import redis_url

log_name = '/src/logs/qa.log'
logging.basicConfig(filename=log_name,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


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

    def embedding(self):
        all_files = glob.glob(self.directory + "/*.csv")
        all_docs = []
        for filename in all_files[0:1]:
            with open(filename, 'r', encoding='utf-8') as f:
                texts = self.text_splitter.split_text(f.read())
                docs = [Document(page_content=t, metadatas={"source": f"{filename}-{i}-pl"}) for i, t in
                        enumerate(texts)]
                all_docs = all_docs + docs
        search_index = Redis.from_documents(all_docs, self.huggingEmbedding, redis_url=redis_url)
        print(search_index)
        return search_index


if __name__ == '__main__':
    em = RedisEmbedding('../it_frame_llms/corpus/audio_txt_clean')
    em.embedding()
