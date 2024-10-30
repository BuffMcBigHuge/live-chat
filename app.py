import asyncio
import os
import time
import warnings
import re

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

from dotenv import load_dotenv
from llm import LanguageModelProcessor
from tts import TextToSpeech
from stt import SpeechToText

load_dotenv()

class ConversationManager:
    def __init__(self):
        self.stt = SpeechToText(model='whisper')
        self.llm = LanguageModelProcessor(type='ollama')
        self.tts = TextToSpeech(model='f5TTS')
        
        self.tts_playing = False
        self.is_listening = True
        self.transcription_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

    async def listen_task(self):
        while self.is_listening:
            def handle_full_sentence(full_sentence):
                if full_sentence:
                    asyncio.create_task(self.transcription_queue.put(full_sentence))
                # Clear the response queue
                self.response_queue.empty()
                self.tts.audio_queue.empty()
            try:
                await self.stt.process(handle_full_sentence)
            except Exception as e:
                print(f"Error in listen_task: {e}")
                await asyncio.sleep(0.1)

    async def process_task(self):
        while self.is_listening:
            try:
                # Get transcription from queue
                transcription = await self.transcription_queue.get()
                
                if "goodbye" in transcription.lower():
                    self.is_listening = False
                    break

                # Process with LLM
                llm_response = self.llm.process(transcription)

                # Remove anything between * characters
                llm_response = re.sub(r'\*.*\*', '', llm_response)
                
                # Put response in queue for TTS
                await self.response_queue.put(llm_response)
                
            except Exception as e:
                print(f"Error in process_task: {e}")
                await asyncio.sleep(0.1)

    async def speak_task(self):
        while self.is_listening:
            try:
                # Get response from queue
                response = await self.response_queue.get()
                
                # Speak the response
                await self.tts.process(text=response)
                
            except Exception as e:
                print(f"Error in speak_task: {e}")
                await asyncio.sleep(0.1)

    async def main(self):
        # Create tasks for listening, processing, and speaking
        tasks = [
            asyncio.create_task(self.listen_task()),
            asyncio.create_task(self.process_task()),
            asyncio.create_task(self.speak_task())
        ]
        
        # Wait for all tasks to complete
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.is_listening = False
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    print(f'Running ConversationManager...')
    manager = ConversationManager()
    asyncio.run(manager.main())
