import json

from flask import Flask, request
from flask_api import status
from langchain import PromptTemplate, ConversationChain
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from .chatbot import Bloom, Profile
import opencc
import ast
from .constants import redis_url

llm = Bloom()
llm.build_extra({'temperature': 1, "top_k": 100, "top_p": 1, "max_new_tokens": 100, "repetition_penalty": 3})

app = Flask(__name__)

converter_t2s = opencc.OpenCC('t2s.json')
converter_s2t = opencc.OpenCC('s2t.json')


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

    profile = Profile(session_id, url=redis_url)
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
    print(pre_template)
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
    print(memory.buffer)
    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        prompt=prompt,
        memory=memory
    )

    r = conversation.predict(input=request_text)
    r = converter_s2t.convert(r)
    response = {"text": r, "session_id": session_id}
    return response, status.HTTP_200_OK


# if __name__ == '__main__':
#     app.run(host='localhost', port=3000)
