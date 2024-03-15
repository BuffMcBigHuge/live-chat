import os
import shutil
import asyncio
import subprocess
import requests
import time
import torch
import subprocess
import wave
import io
import numpy as np
from dotenv import load_dotenv

import edge_tts
from TTS.api import TTS

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextToSpeech:
    language = 'en'
    speaker_wav = None

    def __init__(self):
        tts_mapping = {
            'deepgram': self.deepgramTTS,
            'edgeTTS': self.edgeTTS,
            'coquiTTS': self.coquiTTS
        }
        print("Select the TTS type:")
        for i, tts in enumerate(tts_mapping.keys(), start=1):
            print(f"{i}. {tts}")
        tts_index = int(input("Enter the number of your choice: ")) - 1
        tts_type = list(tts_mapping.keys())[tts_index]
        self.tts_class = tts_mapping.get(tts_type)

        if self.tts_class is None:
            raise ValueError(f'Invalid tts type: {tts_type}')

        if self.tts_class == self.deepgramTTS:
            asyncio.run(self.select_deepgramTTS_voice())
        elif self.tts_class == self.edgeTTS:
            asyncio.run(self.select_edgeTTS_voice())
        elif self.tts_class == self.coquiTTS:
            asyncio.run(self.select_coquiTTS_voice())
        else:
            self.voice = None

    async def select_edgeTTS_voice(self):
        voices = await edge_tts.list_voices()
        print("Select the voice:")
        for i, voice in enumerate(voices, start=1):
            print(f"{i}. {voice['FriendlyName']}")
        voice_index = int(input("Enter the number of your choice: ")) - 1
        self.voice = voices[voice_index]['ShortName']

    async def select_coquiTTS_voice(self):
        # Get a list of .wav files in the ./voices directory
        voices = [f for f in os.listdir('./voices') if f.endswith('.wav')]
        if not voices:
            raise ValueError("No .wav files found in the ./voices directory.")

        print("Select the voice:")
        for i, voice in enumerate(voices, start=1):
            print(f"{i}. {voice}")
        voice_index = int(input("Enter the number of your choice: ")) - 1
        self.voice = os.path.join('./voices', voices[voice_index])
        
    async def select_deepgramTTS_voice(self):
        models = [
            "aura-asteria-en",
            "aura-luna-en",
            "aura-stella-en",
            "aura-athena-en",
            "aura-hera-en",
            "aura-orion-en",
            "aura-arcas-en",
            "aura-perseus-en",
            "aura-angus-en",
            "aura-orpheus-en",
            "aura-helios-en",
            "aura-zeus-en"
        ]

        print("Select the model:")
        for i, model in enumerate(models, start=1):
            print(f"{i}. {model}")
        model_index = int(input("Enter the number of your choice: ")) - 1
        self.voice = models[model_index]
        
    async def process(self, text):
        await self.tts_class(text)

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    async def deepgramTTS(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.voice}&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            if r.status_code != 200:
                raise ValueError(f"Request to Deepgram API failed with status code {r.status_code}. \n\n{r.text}")
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:                        
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f">> TTS ({elapsed_time}ms)")

    async def edgeTTS(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")
        
        start_time = time.time()

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()

        communicate = edge_tts.Communicate(text, self.voice).stream()

        async for chunk in communicate:
            if chunk['type'] == 'audio':
                player_process.stdin.write(chunk['data'])
                player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f">> TTS ({elapsed_time}ms)")

    async def coquiTTS(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")
        
        start_time = time.time()

        player_command = ["ffplay", "-autoexit", "-nodisp", "-"]  
        player_process = subprocess.Popen(
            player_command, 
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        tts_output = tts.tts(
            text=text, 
            speaker_wav=self.voice,
            language=self.language)
        
        tts_output_array = np.array(tts_output)
        int_data = np.int16(tts_output_array * 32767)

        # Create a WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wave_file:
            wave_file.setnchannels(1)  # Mono
            wave_file.setsampwidth(2)  # 16 bits per sample
            wave_file.setframerate(24000)  # Example sample rate
            wave_file.writeframes(int_data.tobytes())
        
        # Write the WAV file's bytes to ffplay's stdin
        buffer.seek(0)  # Go to the beginning of the WAV file in memory
        player_process.stdin.write(buffer.read())
        player_process.stdin.flush()

        #for sample in tts_output:
        #    player_process.stdin.write(sample.tobytes())
        #    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f">> TTS ({elapsed_time}ms)")