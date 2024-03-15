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
global_ndarray = None
silence_threshold=400
silence_ratio=100

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
    def __init__(self):
        stt_mapping = {
            'deepgram': self.deepgram, 
            'whisper': self.whisper
        }
        print("Select the STT type:")
        for i, stt in enumerate(stt_mapping.keys(), start=1):
            print(f"{i}. {stt}")
        stt_index = int(input("Enter the number of your choice: ")) - 1
        stt_type = list(stt_mapping.keys())[stt_index]
        self.stt_class = stt_mapping.get(stt_type)

        if self.stt_class is None:
            raise ValueError(f'Invalid stt type: {stt_type}')
        
    async def process(self, callback):
        await self.stt_class(callback)

    async def deepgram(self, callback):
        global transcript_collector, global_ndarray

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

        print("Listening...")

        def audio_callback(indata, frames, audiotime, status, **kwargs):   
            global global_ndarray, silence_ratio, silence_threshold

            indata_flattened = abs(indata.flatten())

            # concatenate buffers if the global buffer is not empty
            if (global_ndarray is not None):
                global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
            else:
                global_ndarray = indata
                
            # concatenate buffers if the end of the current buffer is not silent
            if (np.average((indata_flattened[-100:-1])) > silence_threshold/15):
                return;
            else:
                local_ndarray = global_ndarray.copy()
                global_ndarray = None
                indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            
            # discard buffers that contain mostly silence
            if (np.asarray(np.where(indata_flattened > silence_threshold)).size < silence_ratio):
                print("Silence...")
            else:
                start_time = time.time()
                model = whisper.load_model('base', device=device) # Can choose another model (https://github.com/openai/whisper)
                result = model.transcribe(indata_transformed, language='en', no_speech_threshold=0.1)
                end_time = time.time()
                elapsed_time = int((end_time - start_time) * 1000)
                print(f">> STT ({elapsed_time}ms)")

                del local_ndarray
                del indata_flattened

                if isinstance(result, dict):
                    # Handle the case where result is a dictionary
                    segments = result.get('segments', [])
                else:
                    # Handle the case where result is an expected object with a 'segments' attribute
                    segments = result.segments

                for segment in segments:
                    if isinstance(segment, dict):
                        # Handle the case where segment is a dictionary
                        text = segment.get('text', '').strip()
                    else:
                        # Handle the case where segment is an expected object with a 'text' attribute
                        text = segment.text.strip() if segment.text else ''

                    if len(text) > 0:
                        transcript_collector.add_part(text)
                        full_sentence = transcript_collector.get_full_transcript()
                        if len(full_sentence.strip()) > 0:
                            full_sentence = full_sentence.strip()
                            print(f"Human: {full_sentence}")
                            callback(full_sentence)  # Call the callback with the full_sentence
                            transcript_collector.reset()
                            transcription_complete.set()  # Signal to stop transcription and exit
        
        with sd.InputStream(samplerate=16000, dtype='int16', channels=1, blocksize=24678, callback=audio_callback):
            try:
                transcription_complete.wait()
                raise sd.CallbackStop
            except sd.CallbackStop:
                return

