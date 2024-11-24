import asyncio
import os
import time
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

from dotenv import load_dotenv
import torch
import sounddevice as sd
import numpy as np
from whisper_online import FasterWhisperASR, OnlineASRProcessor
from silero_vad import FixedVADIterator
import torch.hub

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

BLOCKSIZE = 16000
SAMPLE_RATE = 16000

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

class SpeechToText:
    model = None

    def __init__(self, model=None):
        stt_mapping = {
            'deepgram': self.deepgram, 
            'whisper': self.whisper
        }

        if model is not None:
            self.stt_class = stt_mapping.get(model)
        else:
            print("Select the STT type:")
            for i, stt in enumerate(stt_mapping.keys(), start=1):
                print(f"{i}. {stt}")
            stt_index = int(input("Enter the number of your choice: ")) - 1
            stt_type = list(stt_mapping.keys())[stt_index]
            self.stt_class = stt_mapping.get(stt_type)
        
        if self.stt_class is None:
            raise ValueError(f'Invalid stt type: {stt_type}')
            
        if self.stt_class == self.whisper:
            # Initialize the whisper streaming model
            self.asr = FasterWhisperASR("en", "turbo")
            self.online_processor = OnlineASRProcessor(self.asr)
            self.whole_speech = ""
            
            # Initialize Silero VAD
            vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad'
            )
            self.vad = FixedVADIterator(vad_model, threshold=0.2, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=300, speech_pad_ms=30)

    async def process(self, callback):
       await self.stt_class(callback)

    async def whisper(self, callback):
        global transcript_collector
        main_loop = asyncio.get_running_loop()
        transcription_complete = asyncio.Event()
        
        # Reduce buffer size and timeout for quicker response
        vad_buffer = []
        SPEECH_TIMEOUT = 0.7  # reduced from 1.5 seconds
        MIN_SPEECH_LENGTH = 3  # minimum number of words to consider as valid speech
        last_speech_time = time.time()

        def audio_callback(indata, frames, audiotime, status, **kwargs):
            try:
                nonlocal last_speech_time
                indata_transformed = indata.flatten().astype(np.float32) / 32768.0
                
                # Process with Silero VAD
                vad_result = self.vad(indata_transformed)
                vad_buffer.append(vad_result is not None)
                
                # Keep only last 5 VAD decisions (reduced from 10)
                if len(vad_buffer) > 5:
                    vad_buffer.pop(0)
                
                # Consider speech active if any recent frames contained speech
                is_speech = any(vad_buffer)
                
                if is_speech:
                    last_speech_time = time.time()
                    self.online_processor.insert_audio_chunk(indata_transformed)
                    output = self.online_processor.process_iter()

                    if isinstance(output, tuple) and len(output) == 3 and isinstance(output[2], str):
                        text = output[2].strip()
                        if len(text) > 0:
                            self.whole_speech = text
                            transcript_collector.add_part(text)
                
                # Check for end of speech and process if we have enough content
                elif time.time() - last_speech_time > SPEECH_TIMEOUT and transcript_collector.transcript_parts:
                    full_text = transcript_collector.get_full_transcript()
                    if len(full_text.split()) >= MIN_SPEECH_LENGTH:
                        print(f">> Human: {full_text}")
                        main_loop.call_soon_threadsafe(callback, full_text)
                        transcript_collector.reset()
                        vad_buffer.clear()
                        transcription_complete.set()
                        
            except Exception as e:
                print(f"Error in audio callback: {e}")

        with sd.InputStream(samplerate=SAMPLE_RATE, dtype='int16', channels=1, blocksize=BLOCKSIZE, callback=audio_callback):
            try:
                print("Listening...")
                await transcription_complete.wait()
            except asyncio.CancelledError:
                print("Debug: Transcription cancelled")
                return
            
    async def deepgram(self, callback):
        global transcript_collector

        transcription_complete = asyncio.Event()

        try:
            config = DeepgramClientOptions(options={"keepalive": "true"})
            deepgram: DeepgramClient = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

            dg_connection = deepgram.listen.asynclive.v("1")
            print("Listening...")

            async def on_message(self, result, **kwargs):
                try:
                    sentence = result.channel.alternatives[0].transcript
                    
                    if not result.speech_final:
                        transcript_collector.add_part(sentence)
                    else:
                        transcript_collector.add_part(sentence)
                        full_sentence = transcript_collector.get_full_transcript()
                        
                        if len(full_sentence.strip()) > 0:
                            full_sentence = full_sentence.strip()
                            print(f"Human: {full_sentence}")
                            
                            # Execute callback in the event loop
                            if asyncio.iscoroutinefunction(callback):
                                await callback(full_sentence)
                            else:
                                callback(full_sentence)
                                
                            transcript_collector.reset()
                            transcription_complete.set()
                except Exception as e:
                    print(f"Error in on_message: {e}")

            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                language="en-US",
                encoding="linear16",
                channels=1,
                sample_rate=SAMPLE_RATE,
                endpointing=300,
                smart_format=True,
            )

            await dg_connection.start(options)

            # Open a microphone stream on the default input device
            microphone = Microphone(dg_connection.send, channels=1)
            microphone.start()

            await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

            # Wait for the microphone to close
            microphone.finish()

            # Indicate that we've finished
            await dg_connection.finish()

        except Exception as e:
            print(f"Could not open socket: {e}")
            return
