import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TORCH_LOGS"] = "all"
os.environ["TORCH_SHOW_CPP_STACKTRACES"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="backend:cudaMallocAsync"

import asyncio
import time
import warnings
import re
import sounddevice as sd
import pyaudio

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

from dotenv import load_dotenv
from llm import LanguageModelProcessor
from tts import TextToSpeech
from stt import SpeechToText

load_dotenv()

def get_audio_devices():
    """Get a list of all available audio devices"""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        try:
            device_info = p.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': device_info['name'],
                'maxInputChannels': device_info['maxInputChannels'],
                'maxOutputChannels': device_info['maxOutputChannels'],
                'defaultSampleRate': device_info['defaultSampleRate']
            })
        except Exception as e:
            print(f"Error getting device info for index {i}: {e}")
    
    p.terminate()
    return devices

def get_default_devices():
    """Get the default input and output devices"""
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        
        return {
            'input': {
                'name': default_input['name'],
                'channels': default_input['max_input_channels'],
                'sample_rate': default_input['default_samplerate']
            },
            'output': {
                'name': default_output['name'],
                'channels': default_output['max_output_channels'],
                'sample_rate': default_output['default_samplerate']
            }
        }
    except Exception as e:
        print(f"Error getting default devices: {e}")
        return None

class ConversationManager:
    def __init__(self):
        # Print all available audio devices
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        devices = get_audio_devices()
        '''
        for device in devices:
            print(f"Index: {device['index']}")
            print(f"Name: {device['name']}")
            print(f"Max Input Channels: {device['maxInputChannels']}")
            print(f"Max Output Channels: {device['maxOutputChannels']}")
            print(f"Default Sample Rate: {device['defaultSampleRate']}")
            print("-" * 50)
        '''
        # Print default devices
        default_devices = get_default_devices()
        if default_devices:
            print("\nActive Audio Devices:")
            print("-" * 50)
            print("Default Input Device (Microphone):")
            print(f"Name: {default_devices['input']['name']}")
            print(f"Channels: {default_devices['input']['channels']}")
            print(f"Sample Rate: {default_devices['input']['sample_rate']}")
            print("\nDefault Output Device (Speakers):")
            print(f"Name: {default_devices['output']['name']}")
            print(f"Channels: {default_devices['output']['channels']}")
            print(f"Sample Rate: {default_devices['output']['sample_rate']}")
            print("-" * 50)

        # Add parrot mode selection
        print("\nSelect mode:")
        print("1. Normal (LLM) mode")
        print("2. Parrot mode (repeat speech)")
        while True:
            try:
                mode = int(input("Enter your choice (1 or 2): "))
                if mode in [1, 2]:
                    break
                print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        self.parrot_mode = (mode == 2)
        print(f"\nRunning in {'parrot' if self.parrot_mode else 'normal'} mode...")

        self.stt = SpeechToText(model='deepgram')
        self.llm = None if self.parrot_mode else LanguageModelProcessor(type='ollama')
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

                if "reset" in transcription.lower() and not self.parrot_mode:
                    # Remove history from memory
                    self.llm.reset()
                    continue

                # Process with LLM or parrot directly
                if self.parrot_mode:
                    response = transcription
                else:
                    response = self.llm.process(transcription)
                    # Remove anything between * characters
                    response = re.sub(r'\*.*\*', '', response)
                
                # Put response in queue for TTS
                await self.response_queue.put(response)
                
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
