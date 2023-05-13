import json
import os
import uuid
from urllib.parse import urljoin

import requests
from playsound import playsound

from settings import settings


def text_to_voice(text: str, audio_dir: str) -> str:

    print(f"音声合成スタート: {text}")
    res1 = requests.post(
        urljoin(settings.voicevox_address, "audio_query"),
        params={"text": text, "speaker": 8},
    )

    res2 = requests.post(
        urljoin(settings.voicevox_address, "synthesis"),
        params={"speaker": 8},
        data=json.dumps(res1.json())
    )

    filename = f"{uuid.uuid4()}.wav"
    audio_file = os.path.join(audio_dir, filename)

    with open(audio_file, mode="wb") as f:
        f.write(res2.content)

    print(f"音声合成エンド: {text}")
    
    return audio_file


def play_audio(audio_file: str):
    print("読み上げスタート")
    playsound(audio_file)
    print("読み上げエンド")
