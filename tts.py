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
import threading

from pydub.playback import play
from pydub import AudioSegment
from dotenv import load_dotenv

import edge_tts
from TTS.api import TTS

# import queue
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextToSpeech:
    language = 'en'
    speaker_wav = None
    tts = None
    voice = None
    '''
    chunk_arrival_times = []  # To track arrival times of chunks
    buffer_size = 5
    chunks_queue = queue.Queue()
    '''

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

    async def select_edgeTTS_voice(self):
        voices = await edge_tts.list_voices()
        print("Select the voice:")
        for i, voice in enumerate(voices, start=1):
            print(f"{i}. {voice['FriendlyName']}")

        while True:
            try:
                voice_index = int(input("Enter the number of your choice: ")) - 1
                if voice_index < 0 or voice_index >= len(voices):
                    print("Invalid choice. Please enter a number corresponding to the list of voices.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")

        self.voice = voices[voice_index]['ShortName']

    async def select_coquiTTS_voice(self):
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
                
        self.voice = os.path.join('./voices', voices[voice_index])
        
        # Init tts model
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=device=='cuda').to(device)
        
        '''
        config = XttsConfig()
        config.load_json("./XTTS-v2/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/", use_deepspeed=False)
        # self.model.cuda()
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[self.voice])
        '''

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
        start_time = time.time()

        '''
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")
        player_command = ["ffplay", "-f", "s16le", "-ar", "24000", "-ac", "1", "-nodisp", "-"]
        # player_command = ["ffplay", "-autoexit", "-nodisp", "-"]  
        player_process = subprocess.Popen(
            player_command, 
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        '''

        '''
        chunks = self.model.inference_stream(
            text,
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
            enable_text_splitting=True,
            stream_chunk_size=20,
        )
        '''

        ''' SAVE CHUNKS TO FILE
        wav_chuncks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0}")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
            wav_chuncks.append(chunk)
        wav = torch.cat(wav_chuncks, dim=0)
        torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
        '''

        '''
        threading.Thread(target=self.play_audio)

        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunk: {time.time() - start_time}")

            # Record the arrival time of the chunk
            self.chunk_arrival_times.append(time.time())
            
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

            if isinstance(chunk, list):
                chunk = torch.cat(chunk, dim=0)
            chunk = chunk.clone().detach().cpu().numpy()
            chunk = np.clip(chunk, -1, 1)
            chunk = (chunk * 32767).astype(np.int16)  # Convert to 16-bit PCM

            # Convert numpy array to audio
            audio = AudioSegment(chunk.tobytes(), 
                         frame_rate=24000,
                         sample_width=chunk.dtype.itemsize, 
                         channels=1)

            self.chunks_queue.put(audio)

            # Dynamically adjust buffer size based on the speed chunks are coming in
            self.update_buffer_size_based_on_arrival_rate()

            # Write the chunk's bytes to ffplay's stdin directly without calling .read()
            # player_process.stdin.write(chunk.tobytes())
            # player_process.stdin.flush()
        
        # Wait for all chunks to be played
        while not self.chunks_queue.empty():
            time.sleep(0.1)  # Prevent this loop from using 100% CPU
        
        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f">> TTS ({elapsed_time}ms)")
        '''
        
        '''
        # if player_process.stdin:
        #     player_process.stdin.close()
        # player_process.wait()
        '''

        tts_output = self.tts.tts(
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
        buffer.seek(0)  # Important: rewind the buffer to the beginning
        audio_segment = AudioSegment.from_wav(buffer)
        
        # Play Audio in threading using pydub
        threading.Thread(target=lambda: self.play_audio(audio_segment)).start()

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f">> TTS ({elapsed_time}ms)")

        '''
        # Write the WAV file's bytes to ffplay's stdin
        buffer.seek(0)  # Go to the beginning of the WAV file in memory
        player_process.stdin.write(buffer.read())
        player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()
        '''

    def play_audio(self, wave_file):
        # Play the audio
        play(wave_file)
    
    '''
    def play_audio_chunks(self):
        while True:
            try:
                # Wait until the buffer size is reached
                while self.chunks_queue.qsize() < self.buffer_size:
                    time.sleep(0.1)  # Prevent this loop from using 100% CPU

                # Get the next chunk from the queue
                chunk = self.chunks_queue.get()

                # Convert the chunk to an AudioSegment
                audio = AudioSegment(
                    data=chunk.raw_data,
                    sample_width=chunk.sample_width,
                    frame_rate=chunk.frame_rate,
                    channels=chunk.channels
                )

                # Play the audio
                play(audio)

            except queue.Empty:
                # If the queue is empty, break the loop
                break
    
    def update_buffer_size_based_on_arrival_rate(self):
        if len(self.chunk_arrival_times) < 2:
            return  # Need at least two chunks to calculate a rate

        # Calculate average arrival time between chunks
        arrival_diffs = [self.chunk_arrival_times[i] - self.chunk_arrival_times[i - 1] for i in range(1, len(self.chunk_arrival_times))]
        average_arrival_time = sum(arrival_diffs) / len(arrival_diffs)

        # Adjust buffer size based on average arrival time
        # This is a simplified example; you may want to refine how the buffer size is calculated
        new_buffer_size = max(1, int(average_arrival_time * 2))  # Example adjustment

        print(f"Average arrival time: {average_arrival_time} seconds")
        print(f"New buffer size: {new_buffer_size}")

        # Set the new buffer size
        self.buffer_size = new_buffer_size
    '''