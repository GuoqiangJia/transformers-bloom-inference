import logging
import os
import requests
from functools import partial
from typing import Any, List, Dict, Mapping, Optional

from flask import Flask, request
from flask_api import status
from langchain import PromptTemplate, ConversationChain
from langchain.llms.base import LLM
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from pydantic import BaseModel, Extra, root_validator

from .constants import redis_url
from .constants import HF_ACCELERATE
from .constants import inference_url
from .model_handler.deployment import ModelDeployment
from .utils import (
    ForwardRequest,
    GenerateRequest,
    TokenizeRequest,
    get_exception_response,
    get_num_tokens_to_generate,
    get_torch_dtype,
    parse_bool,
    run_and_log_time,
)

import opencc
import ast
import json

logger = logging.getLogger(__name__)


class QueryID(BaseModel):
    generate_query_id: int = 0
    tokenize_query_id: int = 0
    forward_query_id: int = 0


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


# placeholder class for getting args. gunicorn does not allow passing args to a
# python script via ArgumentParser
class Args:
    def __init__(self) -> None:
        self.deployment_framework = os.getenv("DEPLOYMENT_FRAMEWORK", HF_ACCELERATE)
        self.model_name = os.getenv("MODEL_NAME")
        self.model_class = os.getenv("MODEL_CLASS")
        self.dtype = get_torch_dtype(os.getenv("DTYPE"))
        self.allowed_max_new_tokens = int(os.getenv("ALLOWED_MAX_NEW_TOKENS", 100))
        self.max_input_length = int(os.getenv("MAX_INPUT_LENGTH", 512))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 4))
        self.debug = parse_bool(os.getenv("DEBUG", "false"))


# ------------------------------------------------------
args = Args()
model = ModelDeployment(args, True)
query_ids = QueryID()
app = Flask(__name__)

# ------------------------------------------------------


llm = Bloom()
llm.build_extra({'temperature': 1, "top_k": 100, "top_p": 1, "max_new_tokens": 100, "repetition_penalty": 3})
converter_t2s = opencc.OpenCC('t2s.json')
converter_s2t = opencc.OpenCC('s2t.json')


@app.route("/query_id/", methods=["GET"])
def query_id():
    return query_ids.dict(), status.HTTP_200_OK


@app.route("/tokenize/", methods=["POST"])
def tokenize():
    try:
        x = request.get_json()
        x = TokenizeRequest(**x)

        response, total_time_taken = run_and_log_time(partial(model.tokenize, request=x))

        response.query_id = query_ids.tokenize_query_id
        query_ids.tokenize_query_id += 1
        response.total_time_taken = "{:.2f} msecs".format(total_time_taken * 1000)

        return response.dict(), status.HTTP_200_OK
    except Exception:
        response = get_exception_response(query_ids.tokenize_query_id, args.debug)
        query_ids.tokenize_query_id += 1
        return response, status.HTTP_500_INTERNAL_SERVER_ERROR


@app.route("/generate/", methods=["POST"])
def generate():
    try:
        x = request.get_json()
        x = GenerateRequest(**x)

        x.max_new_tokens = get_num_tokens_to_generate(x.max_new_tokens, args.allowed_max_new_tokens)

        response, total_time_taken = run_and_log_time(partial(model.generate, request=x))

        response.query_id = query_ids.generate_query_id
        query_ids.generate_query_id += 1
        response.total_time_taken = "{:.2f} secs".format(total_time_taken)

        return response.dict(), status.HTTP_200_OK
    except Exception:
        response = get_exception_response(query_ids.generate_query_id, args.debug)
        query_ids.generate_query_id += 1
        return response, status.HTTP_500_INTERNAL_SERVER_ERROR


@app.route("/forward/", methods=["POST"])
def forward():
    try:
        x = request.get_json()
        x = ForwardRequest(**x)

        if len(x.conditioning_text) != len(x.response):
            raise Exception("unequal number of elements in conditioning_text and response arguments")

        response, total_time_taken = run_and_log_time(partial(model.forward, request=x))

        response.query_id = query_ids.forward_query_id
        query_ids.forward_query_id += 1
        response.total_time_taken = "{:.2f} secs".format(total_time_taken)

        return response.dict(), status.HTTP_200_OK
    except Exception:
        response = get_exception_response(query_ids.forward_query_id, args.debug)
        query_ids.forward_query_id += 1
        return response, status.HTTP_500_INTERNAL_SERVER_ERROR


@app.route("/profile/", methods=["POST"])
def profile():
    x = request.get_json()
    x = converter_t2s.convert(str(x))
    x = ast.literal_eval(x)
    session_id = x["session_id"]
    profile = Profile(session_id, url=redis_url)
    profile.set(x)
    return profile.values, status.HTTP_200_OK


@app.route("/chat/", methods=["POST"])
def chat():
    logger.info('enter chat endpoint')
    x = request.get_json()
    x = converter_t2s.convert(str(x))
    x = ast.literal_eval(x)
    session_id = x["session_id"]
    request_text = x["text"]

    if not session_id or not request_text or len(request_text) < 1:
        raise Exception("no session id specified.")

    if not request_text or len(request_text) < 1:
        raise Exception("no chat text specified.")

    request_text = request_text[0]
    logger.info(f'debug info {request_text}')

    profile = Profile(session_id, url=redis_url)
    logger.info(f'debug info {profile}')

    profile_values = json.loads(profile.values.decode("utf-8"))
    profile_express = []
    for p in profile_values:
        if p == 'session_id':
            continue
        v = profile_values[p]
        profile_express.append(f'你的{p}是{v}')

    pre_template = f"你是一个负责和人类聊天的AI虚拟女朋友"
    for p in profile_express:
        pre_template = pre_template + "，" + p

    pre_template = pre_template + "。\n"
    logger.info(f'debug info {pre_template}')
    template = pre_template + """

        {history}
        Human: {input}
        AI:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    history = RedisChatMessageHistory(session_id=session_id, url=redis_url)
    memory = ConversationBufferMemory(memory_key="history", input_key="input", chat_memory=history)
    logger.info(f'debug info {memory.buffer}')
    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        prompt=prompt,
        memory=memory
    )

    r = conversation.predict(input=request_text)
    r = converter_s2t.convert(r)
    response = {"text": r, "session_id": session_id}

    logger.info(f'debug info {response}')
    return response, status.HTTP_200_OK


@app.route("/hello/", methods=["GET"])
def hello():
    logger.info('enter chat hello')
    return {"text": "I am still here."}
