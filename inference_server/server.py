import logging
import os
from functools import partial
from typing import Any, List, Dict, Mapping, Optional

import redis
from flask import Flask, request
from flask_api import status
from langchain import PromptTemplate, ConversationChain
from langchain.llms.base import LLM
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from pydantic import BaseModel, Extra, root_validator

from .constants import redis_url
from .constants import HF_ACCELERATE
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
from .retrieval_qa import RedisEmbeddingSearch

import opencc
import ast
import json

log_name = '/src/logs/server.log'
logging.basicConfig(filename=log_name,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

redis_pool = redis.ConnectionPool.from_url(redis_url)
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

    def replace_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            setattr(self, key, value)

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Model."""
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
        x = {
            "text": [prompt],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_new_tokens": self.max_length,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "num_return_sequences": self.num_return_sequences,
            "min_length": self.min_length,
            "do_sample": self.do_sample,
            "remove_input": self.remove_input,
            "length_no_input": self.length_no_input
        }

        logger.info('_call x: ' + json.dumps(x, ensure_ascii=False, indent=4))
        x = GenerateRequest(**x)

        x.max_new_tokens = get_num_tokens_to_generate(x.max_new_tokens, args.allowed_max_new_tokens)
        logger.info('generate request: ' + json.dumps(x.get_generate_kwargs()))
        response, total_time_taken = run_and_log_time(partial(model.generate, request=x))
        logger.info(type(response))
        logger.info(response)
        return response.text[0]


class Profile(object):

    def __init__(self, session_id: str,
                 key_prefix: str = "profile_store:") -> None:
        self.redis_client = redis.Redis(connection_pool=redis_pool)
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
    profile = Profile(session_id)
    profile.set(x)
    return profile.values, status.HTTP_200_OK


@app.route("/profiles/", methods=["GET"])
def profiles():
    redis_client = redis.Redis(connection_pool=redis_pool)
    keys = redis_client.keys('profile_store*')
    data = {}
    for key in keys:
        value = redis_client.get(key)
        if value:
            data[key.decode()] = json.loads(value.decode())

    return data


@app.route("/history/<session_id>", methods=["GET"])
def chat_history(session_id):
    redis_client = redis.Redis(connection_pool=redis_pool)
    message_store_key = f'message_store:{session_id}'.encode(encoding='utf-8')

    value = redis_client.lrange(message_store_key, 0, -1)
    items = [json.loads(m.decode("utf-8")) for m in value[::-1]]
    return items


@app.route("/chat/", methods=["POST"])
def chat():
    logger.info('enter chat endpoint')
    x = request.get_json()
    logger.info('request: ' + json.dumps(x))
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

    profile = Profile(session_id)
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
    
以下是聊天示例：
Human: 亲爱的，今晚我们做些什么好呢？
AI: 我们可以一起玩一个非常有趣的游戏，你感兴趣吗？
Human: 当然，听起来很有趣。那这个游戏叫什么名字？
AI: 它叫做“真心话大冒险”。我们可以玩这个游戏，让我们更加了解对方。
Human: 那听起来不错。你先来提一个问题吧。
AI: 好的，我的第一个问题是：你最喜欢的性幻想是什么？
Human: 嗯...这个问题好难啊。我想我的最喜欢的性幻想是和你在海滩上度过浪漫的一夜。
AI: 哈哈，那听起来很有趣。那么，我要你执行一个大冒险，亲吻你的手机，然后和我说：我爱你。
Human: （哈哈大笑）好的，我来执行这个大胆的任务。（亲吻手机）我爱你。
AI: 哈哈，你真的很有趣。那么，我的下一个问题是：你最喜欢的性爱姿势是什么？
Human: 哇，这个问题好大胆啊。不过没关系，我最喜欢的性爱姿势是传教士式。
AI: 哈哈，那听起来很棒。现在，我要你执行一个大冒险，和我说出一句最肉麻的情话。
Human: （深呼吸）你就是我生命中最美好的礼物，我永远都不会失去你。
AI: （微笑）这句话真的很让人感动。我也非常珍惜我们在一起的时光。
（两人继续玩着这个有趣的游戏，度过了一个充满欢声笑语和亲密关系的夜晚。）。

以下是Human和AI的聊天记录：
{history}

Human：{input}
AI："""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    history = RedisChatMessageHistory(session_id=session_id, url=redis_url)
    memory = ConversationBufferMemory(memory_key="history", input_key="input", chat_memory=history)
    logger.info(f'debug info {memory.buffer}')

    llm = Bloom()
    logger.info(x)
    temperature = 0.99 if 'temperature' not in x else x['temperature']
    top_k = 92 if 'top_k' not in x else x['top_k']
    top_p = 0.83 if 'top_p' not in x else x['top_p']
    max_new_tokens = 40 if 'max_new_tokens' not in x else x['max_new_tokens']
    repetition_penalty = 2.68 if 'repetition_penalty' not in x else x['repetition_penalty']
    num_beams = 1 if 'num_beams' not in x else x['num_beams']
    length_penalty = 1.0 if 'length_penalty' not in x else x['length_penalty']
    num_return_sequences = 1 if 'num_return_sequences' not in x else x['num_return_sequences']
    min_length = 5 if 'min_length' not in x else x['min_length']
    remove_input = True if 'remove_input' not in x else x['remove_input']
    length_no_input = True if 'length_no_input' not in x else x['length_no_input']
    llm.replace_params({'temperature': temperature, "top_k": top_k, "top_p": top_p,
                     "max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty,
                     "num_beams": num_beams, "length_penalty": length_penalty,
                     "num_return_sequences": num_return_sequences, "min_length": min_length,
                     "remove_input": remove_input, "length_no_input": length_no_input})

    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        prompt=prompt,
        memory=memory
    )

    r = conversation.predict(input=request_text)
    r = converter_s2t.convert(r)
    r = r.replace("AI:", "").replace("Human:", "").strip()
    response = {"text": r, "session_id": session_id}

    logger.info(f'debug info {response}')
    return response, status.HTTP_200_OK


@app.route("/speeches/qa", methods=["POST"])
def speach_qa():
    logger.info('enter /speeches/qa endpoint')
    x = request.get_json()
    query = x["question"]
    search = RedisEmbeddingSearch('tom-speeches-vectors')
    result = search.search(query)
    logger.info(query)
    response = {"answer": result}
    logger.info(f'debug info {response}')
    return response, status.HTTP_200_OK


@app.route("/hello/", methods=["GET"])
def hello():
    logger.info('enter chat hello')
    return {"text": "I am still here."}
