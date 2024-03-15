import asyncio
import os
import time
from dotenv import load_dotenv
from llm import LanguageModelProcessor
from tts import TextToSpeech
from stt import SpeechToText

load_dotenv()

class ConversationManager:
    transcription_response = ""
    llm = LanguageModelProcessor()
    tts = TextToSpeech()
    stt = SpeechToText()

    def __init__(self):
        pass

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            # Listening with STT
            await self.stt.process(handle_full_sentence)

            if "goodbye" in self.transcription_response.lower():
                break

            # Processing with LLM
            llm_response = self.llm.process(self.transcription_response)

            # Speaking with TTS
            await self.tts.process(text=llm_response)

            self.transcription_response = ""

if __name__ == "__main__":
    print(f'Running ConversationManager...')
    manager = ConversationManager()
    asyncio.run(manager.main())