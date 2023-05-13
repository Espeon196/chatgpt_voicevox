import concurrent.futures
import queue
import tempfile

from chat import chat
from voicevox import text_to_voice, play_audio
from settings import settings


if __name__ == "__main__":
    audio_queue = queue.Queue()

    with tempfile.TemporaryDirectory() as audio_dir:

        while True:
            prompt = input("話したいことを書いてね: ")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

                g = chat(prompt)
                text = ""
                for token in g:
                    text += token

                    if token in ['。', '？', '！', '.']:
                        audio_file = executor.submit(text_to_voice, text, audio_dir).result()
                        audio_queue.put(audio_file)
                        text = ""
                    
                    if not audio_queue.empty():
                        audio_file = audio_queue.get()
                        executor.submit(play_audio, audio_file)
