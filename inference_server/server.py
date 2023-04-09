import logging
import os
from functools import partial

from flask import Flask, request
from flask_api import status
from langchain import PromptTemplate, ConversationChain
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from pydantic import BaseModel

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

from .chatbot import Bloom, Profile
import opencc
import ast
import json

logger = logging.getLogger(__name__)

class QueryID(BaseModel):
    generate_query_id: int = 0
    tokenize_query_id: int = 0
    forward_query_id: int = 0


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
    return {"text": "I am still here."}
