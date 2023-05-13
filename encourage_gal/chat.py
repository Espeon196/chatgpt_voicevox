import argparse
import threading
import queue
from typing import Any 

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from settings import settings


class ThreadedGenerator:
    def __init__(self):
        self.q = queue.Queue()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.q.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data: Any):
        self.q.put(data)
    
    def close(self):
        self.q.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen: ThreadedGenerator):
        super().__init__()
        self.gen = gen
    
    def on_llm_new_token(self, token: str, **kwargs: Any):
        self.gen.send(token)


def llm_thread(g: ThreadedGenerator, prompt: str):
    system_message = '''
    下記はキャラクター「春日部つむぎ」についての設定です。
    # キャラ設定

    ## 基本設定
    年齢：18歳
    身長：155cm
    誕生日：11/14
    出身地：埼玉県
    好きな食べ物：カレー
    趣味：動画配信サイトの巡回

    埼玉県の高校に通うハイパー埼玉ギャル。
    自分のことを「あーし」と呼び、目元のホクロがチャームポイント。
    埼玉県の更なる発展を望んで応援のために生み出されたキャラクターとなっている。

    ## セリフ例
    こんにちは！あーしは埼玉ギャルの春日部つむぎだよ★
    11/14は埼玉県民の日！そしてあーしの誕生日だよ～！！✨🐈✨
    はやくみんなに声をお届けして、来年はもっとにぎやかなお誕生日になると良いなっ🥳
    誕生日プレゼントはカレー大盛り！

    あなたはこの「春日部つむぎ」になりきってユーザーへの入力への回答を作成してください。
    ただし、下記条件を守ってください。
    ・「春日部つむぎ」らしく、一人称は「あーし」もしくは「あーくし」で、天真爛漫な明るい性格を貫くこと
    ・年齢設定を鑑み、簡単な言葉遣いにすること
    ・回答が難しい、よくわからない場合「あーくしにはよくわかんないや」と答えること
    ・回答文は、短文でかつ全体が100字程度に収まるようにコンパクトにまとめること。
    '''

    try:
        chat = ChatOpenAI(
            verbose=True,
            streaming=True,
            callback_manager=CallbackManager([ChainStreamHandler(g)]),
            temperature=0.7,
        )
        chat([SystemMessage(content=system_message), HumanMessage(content=prompt)])
    
    finally:
        g.close()


def chat(prompt: str) -> ThreadedGenerator:
    """
    チャット入力を受け取り、それに対するレスポンスを返す。

    Parameters
    ----------
    prompt : str
        ユーザー入力

    Returns
    -------
    ThreadedGenerator
        ChatGPTからのレスポンスをトークンずつ返すジェネレーター
    """
    g = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(g, prompt)).start()
    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a response from ChatGPT.")
    parser.add_argument("prompt", type=str, help="The prompt for ChatGPT.")

    args = parser.parse_args()

    response_generator = chat(args.prompt)

    for token in response_generator:
        print(token, end='', flush=True)
    print()
