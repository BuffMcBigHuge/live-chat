import os
import shutil
import asyncio
import subprocess
import requests
import time
import torch
import subprocess
import numpy as np
import threading
import re
import sys

# Add .\venv\Lib\site-packages\f5_tts
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'F5-TTS')):
    print("Cloning F5-TTS repository...")
    subprocess.run(
    [
        "git", "clone", "--depth", "1",
        "https://github.com/SWivid/F5-TTS",
        "F5-TTS"
    ],
    cwd=os.path.dirname(__file__)
    )
sys.path.append(os.path.join(os.path.dirname(__file__), 'F5-TTS', 'src'))
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models/F5TTS_Base')):
    # Download model file
    print("Downloading F5-TTS model file...")
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models/F5TTS_Base'), exist_ok=True)
    url = 'https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.safetensors?download=true'
    response = requests.get(url)
    with open(os.path.join(os.path.dirname(__file__), 'models/F5TTS_Base/model_1200000.safetensors'), 'wb') as f:
        f.write(response.content)

from pydub.playback import play
from pydub import AudioSegment
from dotenv import load_dotenv

import edge_tts

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextToSpeech:
    language = 'en'
    tts_class = None
    voice = None
    voice_ref_audio = None
    voice_ref_text = None
    model = None

    # Add a threading lock for audio playback
    audio_lock = threading.Lock()
    audio_queue = asyncio.Queue()  # Add queue for audio segments
    is_playing = False
    play_task = None

    def __init__(self, model=None, voice=None):
        tts_mapping = {
            'deepgram': self.deepgramTTS,
            'edgeTTS': self.edgeTTS,
            'f5TTS': self.f5TTS
        }

        # Select the TTS model
        if (model is not None):
            self.tts_class  = tts_mapping.get(model)
        else:
            print("Select the TTS model:")
            for i, tts in enumerate(tts_mapping.keys(), start=1):
                print(f"{i}. {tts}")
            tts_index = int(input("Enter the number of your choice: ")) - 1
            tts_type = list(tts_mapping.keys())[tts_index]
            self.tts_class = tts_mapping.get(tts_type)

        if self.tts_class is None:
            raise ValueError(f'Invalid tts model: {tts_type}')

        # Select the voice
        if (voice is not None):
            self.voice = voice
        elif self.tts_class == self.deepgramTTS:
            asyncio.run(self.select_deepgramTTS_voice())
        elif self.tts_class == self.edgeTTS:
            asyncio.run(self.select_edgeTTS_voice())
        elif self.tts_class == self.f5TTS:
            asyncio.run(self.select_f5TTS_voice())

    async def select_edgeTTS_voice(self):
        voices = await edge_tts.list_voices()
        print("Select the voice:")
        english_voices = [voice for voice in voices if voice['Locale'].startswith('en-')]

        for i, voice in enumerate(english_voices, start=1):
            print(f"{i}. f'{voice['FriendlyName']} ({voice['ShortName']})'")

        while True:
            try:
                voice_index = int(input("Enter the number of your choice: ")) - 1
                if voice_index < 0 or voice_index >= len(english_voices):
                    print("Invalid choice. Please enter a number corresponding to the list of voices.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")

        self.voice = english_voices[voice_index]['ShortName']
    
    async def select_deepgramTTS_voice(self):
        voices = [
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

        print("Select the voice:")
        for i, voice in enumerate(voices, start=1):
            print(f"{i}. {voice}")

        while True:
            try:
                voice_index = int(input("Enter the number of your choice: ")) - 1
                if voice_index < 0 or voice_index >= len(voices):
                    print("Invalid choice. Please enter a number corresponding to the list of voices.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")

        self.voice = voices[voice_index]
    
    async def select_f5TTS_voice(self):
        # Select the voice
        self.voice = self.select_local_voice()
        
        # Ensure preprocess_ref_audio_text returns exactly two elements
        ref_audio, ref_text = preprocess_ref_audio_text(
            './voices/' + self.voice,
            "",
            device=device
        )

        print(f"Ref audio: {ref_audio}")
        print(f"Ref text: {ref_text}")
        
        # Assign the correct values
        self.voice_ref_audio = ref_audio
        self.voice_ref_text = ref_text

        # Load F5TTS model
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16,
            ff_mult=2, text_dim=512, conv_layers=4
        )
        ckpt_file = str('./models/F5TTS_Base/model_1200000.safetensors')
        vocab_file = os.path.join('./F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt')
        
        # Ensure the model is loaded correctly
        self.model = load_model(model_cls, model_cfg, ckpt_file, vocab_file)
        if self.model is None:
            raise ValueError("Failed to load the F5TTS model.")

    async def process(self, text):
        await self.tts_class(text)

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def select_local_voice(self):
        # Get a list of .wav files in the ./voices directory
        voices = [f for f in os.listdir('./voices') if f.endswith('.wav')]
        if not voices:
            raise ValueError("No .wav files found in the ./voices directory.")

        print("Select the voice:")
        for i, voice in enumerate(voices, start=1):
            print(f"{i}. {voice}")
        
        while True:
            try:
                voice_index = int(input("Enter the number of your choice: ")) - 1
                if voice_index < 0 or voice_index >= len(voices):
                    print("Invalid choice. Please enter a number corresponding to the list of voices.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"Selected voice: {voices[voice_index]}")

        return voices[voice_index]

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

        with self.audio_lock:  # Ensure only one audio plays at a time
            player_process = subprocess.Popen(
                player_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.player_process = player_process  # Store the process for stopping

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

        with self.audio_lock:  # Ensure only one audio plays at a time
            player_process = subprocess.Popen(
                player_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.player_process = player_process  # Store the process for stopping

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

    async def play_queue(self):
        while True:
            try:
                # Wait for the next audio segment
                audio_segment = await self.audio_queue.get()
                self.is_playing = True
                # Remove the lock since we're already handling synchronization via the queue
                play(audio_segment)
                self.is_playing = False
                # Signal that we've finished processing this queue item
                self.audio_queue.task_done()
            except Exception as e:
                print(f"Error in play_queue: {e}")

    async def process_chunk(self, text):
        """Process a single chunk of text into audio"""
        if not text.strip():  # Skip empty chunks
            return
        
        audio, final_sample_rate, _ = infer_process(
            self.voice_ref_audio, self.voice_ref_text, text, self.model, device=device,
        )

        # Process audio
        final_wave = audio
        max_val = np.max(np.abs(final_wave))
        if max_val > 0:
            final_wave = final_wave / max_val

        final_wave = (final_wave * 32767).astype(np.int16)
        sample_width = final_wave.dtype.itemsize

        # Convert to AudioSegment
        audio_segment = AudioSegment(
            final_wave.tobytes(),
            frame_rate=final_sample_rate,
            sample_width=sample_width,
            channels=1
        )
        audio_segment = audio_segment - 10  # Reduce volume

        return audio_segment

    async def f5TTS(self, text):
        # Start the queue player if not already running
        if not self.play_task:
            self.play_task = asyncio.create_task(self.play_queue())

        # Split text into chunks at punctuation, aiming for ~50 characters per chunk
        def chunk_text(text, target_length=50):
            chunks = []
            current_chunk = ""
            
            # Split at sentence boundaries first
            sentences = re.split(r'([.,;?!-])', text)
            
            for i in range(0, len(sentences), 2):
                part = sentences[i]
                punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
                
                if len(current_chunk) + len(part) + 1 <= target_length:
                    current_chunk += part + punctuation
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part + punctuation
            
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            return [chunk for chunk in chunks if chunk.strip()]

        chunks = chunk_text(text)

        try:
            # Process chunks concurrently and enqueue each result immediately
            for chunk in chunks:
                if chunk:
                    audio_data = await self.process_chunk(chunk)
                    await self.audio_queue.put(audio_data)

        except Exception as e:
            print(f"F5TTS inference failed: {str(e)}")

    def stop_playback(self):
         # Stop audio queue and playback
         pass