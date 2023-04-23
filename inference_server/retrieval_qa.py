import glob
import logging
from abc import ABC, abstractmethod

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from .redis_fix import Redis
from .constants import redis_url

log_name = '/src/logs/server.log'
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
        self.index_name = 'tom-speeches-vectors'

    def embedding(self):
        all_files = glob.glob(self.directory + "/*.csv")
        df_list = []
        for filename in all_files:
            logger.info('Handling ' + filename)
            with open(filename, 'r', encoding='utf-8') as f:
                df = pd.read_csv(filename, index_col=None, header=0, usecols=['id', 'fileName', 'audioText'])
                df_list.append(df)
        df = pd.concat(df_list, axis=0, ignore_index=True).sort_values('id')

        all_docs = []
        df = df.head(10)
        for index, row in df.iterrows():
            title = row['fileName']
            audio_text = title + '.' + row['audioText']
            texts = self.text_splitter.split_text(audio_text)
            docs = [Document(page_content=t, metadatas={"source": f"{title}-{i}-pl"}) for i, t in
                    enumerate(texts)]
            logger.info('Chunk size ' + str(len(docs)))

            all_docs = all_docs + docs

        search_index = Redis.from_documents(documents=all_docs, embedding=self.huggingEmbedding,
                                            redis_url=redis_url, index_name=self.index_name)
        return search_index

    def embedding_test(self):
        s = """
Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
        """
        title = 'test'
        all_docs = [Document(page_content=s, metadatas={"source": f"{title}-{0}-pl"})]
        search_index = Redis.from_documents(documents=all_docs, embedding=self.huggingEmbedding,
                                            redis_url=redis_url, index_name='test_index')
        return search_index


class EmbeddingSearch(ABC):

    def __init__(self, index_name: str) -> None:
        super().__init__()
        self.index_name = index_name

    @abstractmethod
    def search(self, query: str):
        """embedding all files under the directory"""


class RedisEmbeddingSearch(EmbeddingSearch):

    def __init__(self, index_name: str) -> None:
        super().__init__(index_name)
        self.huggingEmbedding = HuggingFaceInstructEmbeddings()

    def search(self, query: str):
        rds = Redis.from_existing_index(embedding=self.huggingEmbedding, redis_url=redis_url,
                                        index_name=self.index_name)
        results = rds.similarity_search(query)
        if not results:
            return ''

        return results[0].page_content


# if __name__ == '__main__':
#     search = RedisEmbeddingSearch('tom-speeches-vectors')
#     search.search('')
# em = RedisEmbedding('../it_frame_llms/corpus/audio_summary_pegasuslarge')
# em.embedding()
# print(em.search("What did the president say about Ketanji Brown Jackson", 'test_index'))

