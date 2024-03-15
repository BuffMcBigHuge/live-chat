# LIVE-CHAT

LIVE-CHAT is a voice-based conversational assistant application that uses Speech-to-Text (STT), Large Language Models (LLMs) and Text-to-Speech (TTS) to chat in your terminal. It's designed to simulate a live conversation with short, conversational responses. Now with enhanced TTS speaker selection and Language Model (LLM) selection features.

## Features

- Text-to-Speech (TTS) support with multiple providers: ([Microsoft Edge TTS](https://github.com/rany2/edge-tts), [Deepgram.com](https://deepgram.com/product/text-to-speech), [Coqui XTTSv2 (Offline)](https://huggingface.co/coqui/XTTS-v2). Now includes the ability to select your preferred TTS speaker.
- Language model processing for conversational responses. Now includes the ability to select your preferred Language Model (LLM), with support with multiple providers: [Groq](https://groq.com/), [OpenAI API](https://openai.com/blog/openai-api), [Ollama (Offline)](https://github.com/ollama/ollama).
- Speech-to-Text (STT) support with multiple providers: ([Deepgram.com](https://deepgram.com/product/speech-to-text), [Whisper (Offline)](https://github.com/openai/whisper)). You can put audio files in `/voices` for custom cloning with Coqui.
- Enhanced user customization options for a more personalized experience.

## Setup

1. Clone the repository
2. Create a Conda environment `conda create -n live-agent python=3.11` and activate it `conda activate live-agent`
3. Install the required Python packages by running `pip install -r requirements.txt`
4. Set up your environment variables in a `.env` file. You'll need to provide your API keys for the TTS and STT services. You can also use the offline modes without any API keys, however you will have to install and configure Ollama.
5. Run the application with `python app.py`

## Usage

When you run the application, you'll be prompted to enter your preferred TTS and STT providers. You can now also select your preferred TTS speaker and Language Model (LLM). After that, the application will start a conversation. You can stop the conversation by saying "goodbye".

The fastest combination of tools that I have found is STT using Deepgram, LLM with Groq, and TTS with Deepgram.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.