import asyncio
import os
import time
from dotenv import load_dotenv
import torch
import sounddevice as sd
import numpy as np
import whisper
import threading

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

BLOCKSIZE=16000
SAMPLE_RATE=16000

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
    global_ndarray = None
    silence_threshold=400 # should be set to the lowest sample amplitude that the speech in the audio material has
    silence_ratio=50 # number of samples in one buffer that are allowed to be higher than threshold
    no_speech_threshold=0.1
    silence_duration_threshold = 1  # seconds of silence before processing
    silence_counter = 0  # counter for continuous silence
    model = None

    def __init__(self, model=None):
        stt_mapping = {
            'deepgram': self.deepgram, 
            'whisper': self.whisper
        }

        #   Select the STT type:
        if (model is not None):
            self.stt_class  = stt_mapping.get(model)
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
            # Can choose another model (https://github.com/openai/whisper)
            self.model = whisper.load_model('turbo', device=device)
        
    async def process(self, callback):
        await self.stt_class(callback)

    async def deepgram(self, callback):
        global transcript_collector

        transcription_complete = asyncio.Event()  # Event to signal transcription completion

        try:
            # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
            config = DeepgramClientOptions(options={"keepalive": "true"})
            deepgram: DeepgramClient = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

            dg_connection = deepgram.listen.asynclive.v("1")
            print ("Listening...")

            async def on_message(self, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                
                if not result.speech_final:
                    transcript_collector.add_part(sentence)
                else:
                    # This is the final part of the current sentence
                    transcript_collector.add_part(sentence)
                    full_sentence = transcript_collector.get_full_transcript()
                    # Check if the full_sentence is not empty before printing
                    if len(full_sentence.strip()) > 0:
                        full_sentence = full_sentence.strip()
                        print(f"Human: {full_sentence}")
                        callback(full_sentence)  # Call the callback with the full_sentence
                        transcript_collector.reset()
                        transcription_complete.set()  # Signal to stop transcription and exit

            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                language="en-US",
                encoding="linear16",
                channels=1,
                sample_rate=16000,
                endpointing=300,
                smart_format=True,
            )

            await dg_connection.start(options)

            # Open a microphone stream on the default input device
            microphone = Microphone(dg_connection.send)
            microphone.start()

            await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

            # Wait for the microphone to close
            microphone.finish()

            # Indicate that we've finished
            await dg_connection.finish()

        except Exception as e:
            print(f"Could not open socket: {e}")
            return

    async def whisper(self, callback):
        global transcript_collector

        transcription_complete = threading.Event()

        def audio_callback(indata, frames, audiotime, status, **kwargs):            
            indata_flattened = abs(indata.flatten())
            
            # Check if the number of samples above the threshold is less than the ratio
            if (np.asarray(np.where(indata_flattened > self.silence_threshold)).size < self.silence_ratio):
                self.silence_counter += 1  # increment the silence counter
                print(f"Silence {self.silence_counter}...")
            else:
                # If the number of samples above the threshold is greater than the ratio, reset the silence counter
                if (self.global_ndarray is not None):
                    self.global_ndarray = np.concatenate((self.global_ndarray, indata), dtype='int16')
                else:
                    self.global_ndarray = indata
                
            # If the silence counter exceeds the silence duration threshold, process the audio
            if (self.silence_counter > self.silence_duration_threshold * (SAMPLE_RATE / frames) and self.global_ndarray is not None):
                self.silence_counter = 0

                local_ndarray = self.global_ndarray.copy()
                self.global_ndarray = None
                indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
                indata_transformed_tensor = torch.tensor(indata_transformed)

                start_time = time.time()
                result = self.model.transcribe(indata_transformed_tensor, language='en', no_speech_threshold=self.no_speech_threshold)
                end_time = time.time()
                elapsed_time = int((end_time - start_time) * 1000)
                print(f">> STT ({elapsed_time}ms)")

                if result['text'] != "":
                    print(f"Human: {result['text']}")
                    callback(result['text'])  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

                del local_ndarray
                del indata_flattened

        with sd.InputStream(samplerate=SAMPLE_RATE, dtype='int16', channels=1, blocksize=BLOCKSIZE, callback=audio_callback):
            try:
                print("Listening...")
                transcription_complete.wait()
                raise sd.CallbackStop
            except sd.CallbackStop:
                return

