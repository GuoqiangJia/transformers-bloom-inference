import json
import logging
from typing import List, Optional, Any, Mapping, Dict

import requests
from langchain import ConversationChain, PromptTemplate
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from pydantic import BaseModel, root_validator
from pydantic import Extra
from constants import redis_url
from constants import inference_url

logger = logging.getLogger(__name__)


class Bloom(LLM, BaseModel):
    """Wrapper around BLOOM large language models.
    """

    client: Any  #: :meta private:
    model_name: str = "Bloom-mt0-xxl-mt"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    min_length: int = 1
    """The minimum number of tokens to generate in the completion."""
    max_length: int = 256
    """The maximum number of tokens to generate in the completion."""
    length_no_input: bool = True
    """Whether min_length and max_length should include the length of the input."""
    remove_input: bool = True
    """Remove input text from API response"""
    remove_end_sequence: bool = True
    """Whether or not to remove the end sequence token."""
    bad_words: List[str] = []
    """List of tokens not allowed to be generated."""
    top_p: int = 1
    """Total probability mass of tokens to consider at each step."""
    top_k: int = 50
    """The number of highest probability tokens to keep for top-k filtering."""
    repetition_penalty: float = 1.0
    """Penalizes repeated tokens. 1.0 means no penalty."""
    length_penalty: float = 1.0
    """Exponential penalty to the length."""
    do_sample: bool = True
    """Whether to use sampling (True) or greedy decoding."""
    num_beams: int = 1
    """Number of beams for beam search."""
    early_stopping: bool = False
    """Whether to stop beam search at num_beams sentences."""
    num_return_sequences: int = 1
    """How many completions to generate for each prompt."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling NLPCloud API."""
        return {
            "temperature": self.temperature,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "length_no_input": self.length_no_input,
            "remove_input": self.remove_input,
            "remove_end_sequence": self.remove_end_sequence,
            "bad_words": self.bad_words,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
            "num_return_sequences": self.num_return_sequences,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "bloom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        data = {
            "text": [prompt],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_new_tokens": self.max_length,
            "repetition_penalty": self.repetition_penalty
        }

        response = requests.post(inference_url, json=data)
        if response.status_code == 200:
            result = json.loads(response.content)['text']
            if len(result) > 0:
                return result[0]
        return ''


class Profile(object):

    def __init__(self, session_id: str,
                 url: str = redis_url,
                 key_prefix: str = "profile_store:") -> None:
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        try:
            self.redis_client = redis.Redis.from_url(url=url)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

        self.session_id = session_id
        self.key_prefix = key_prefix

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

    @property
    def values(self):
        return self.redis_client.get(self.key)

    def set(self, item) -> None:
        """Append the message to the record in Redis"""
        self.redis_client.set(self.key, json.dumps(item))

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.key)


if __name__ == '__main__':
    llm = Bloom()
    llm.build_extra({'temperature': 1, "top_k": 100, "top_p": 1, "max_new_tokens": 100, "repetition_penalty": 3})

    name = "小丽"
    height = "1.70米"
    cup = "C罩杯"
    birthday = "1984年10月3日"
    character = "活泼可爱的"

    pre_template = f"你是一个负责和人类聊天的AI虚拟女朋友，你的名字叫做{name}，你的身高是{height}，你的胸围是{cup}，你的生日是{birthday}，你的性格是{character}。\n"

    template = pre_template + """
    
    {history}
    Human: {input}
    AI:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    history = RedisChatMessageHistory(session_id='chat-0002', url=redis_url)
    memory = ConversationBufferMemory(memory_key="history", input_key="input", chat_memory=history)
    print(memory.buffer)
    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        prompt=prompt,
        memory=memory
    )

    r = conversation.predict(input="好的，昨天我去了哪里啊？")
    print(r)
