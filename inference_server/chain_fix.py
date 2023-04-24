from typing import List, Any, Tuple

from langchain import BasePromptTemplate, LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.schema import Document, BaseLanguageModel
from langchain.chains.qa_with_sources.map_reduce_prompt import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)

from inference_server.logger_factory import LoggerFactory

logger = LoggerFactory.get_logger(__name__, log_level="INFO")


class PassStuffDocumentsChain(StuffDocumentsChain):
    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        results = '\n'
        for i, d in enumerate(docs):
            results = results + str(i + 1) + ". " + d.page_content + '\n'

        logger.info('Answer: ' + results)
        return results, {}


class PassRetrievalQAWithSourcesChain(RetrievalQAWithSourcesChain):

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
                 question_prompt: BasePromptTemplate = QUESTION_PROMPT,
                 combine_prompt: BasePromptTemplate = COMBINE_PROMPT, **kwargs: Any) -> BaseQAWithSourcesChain:
        """Construct the chain from an LLM."""
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        combine_results_chain = PassStuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        combine_document_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            combine_document_chain=combine_results_chain,
            document_variable_name="context",
        )
        return cls(
            combine_documents_chain=combine_document_chain,
            **kwargs,
        )
