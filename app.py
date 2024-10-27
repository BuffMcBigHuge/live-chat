import asyncio
import os
import time
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

from dotenv import load_dotenv
from llm import LanguageModelProcessor
from tts import TextToSpeech
from stt import SpeechToText

load_dotenv()

class ConversationManager:
    transcription_response = ""

    # Init
    # stt = SpeechToText()
    # llm = LanguageModelProcessor()
    # tts = TextToSpeech()

    stt = SpeechToText(model='whisper')
    llm = LanguageModelProcessor(type='ollama')
    tts = TextToSpeech(model='f5TTS')

    def __init__(self):
        self.tts_playing = False

    async def stop_tts(self):
        if self.tts_playing:
            self.tts.stop_playback()  # Assuming you add a stop_playback method in TTS
            self.tts_playing = False

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Init
        llm_response = self.llm.process('Hello! Who are you?')
        await self.tts.process(text=llm_response)
        self.tts_playing = True

        while True:
            # Listening with STT
            await self.stt.process(handle_full_sentence)

            if self.transcription_response:
                await self.stop_tts()  # Stop TTS if new speech is detected

                if "goodbye" in self.transcription_response.lower():
                    break

                # Processing with LLM
                llm_response = self.llm.process(self.transcription_response)

                # Speaking with TTS
                await self.tts.process(text=llm_response)
                self.tts_playing = True

                self.transcription_response = ""

if __name__ == "__main__":
    print(f'Running ConversationManager...')
    manager = ConversationManager()
    asyncio.run(manager.main())
