# Voice-enabled chatbot

## Overview
This project integrates a Large Language Model (LLM)-based chatbot with real-time voice capabilities. The system uses advanced speech-to-text (STT) and text-to-speech (TTS) technologies to enable seamless conversational interactions through voice.

## Features
- **Real-Time Speech Recognition**: Uses WebRTC VAD for speech activity detection and Whisper for transcription.
- **LLM Chatbot Integration**: Processes transcribed text through a Large Language Model (via Groq or Ollama) for intelligent responses.
- **Voice Synthesis**: Converts LLM responses into speech for a complete voice-based conversational experience.
- **Prolonged Silence Detection**: Automatically exits conversations when no speech is detected for a specified duration.

## Tech Stack
- **Python**
- **WebRTC VAD** for speech activity detection
- **OpenAI Whisper** or Hugging Face Transformers for transcription
- **Text-to-Speech (TTS)** gTTs or ParlerTTS
- **LLM**: Llama3-8b via Groq API or Qwen 2.5 1.5b via Ollama

## Prerequisites
1. **Python Environment**: Python 3.8+
2. **Libraries**:
   - `webrtcvad`
   - `numpy`
   - `transformers`
   - `gTTS`
3. **Audio Hardware**: A working microphone and speakers

## Setup
1. Clone the repository:
```bash
git clone https://github.com/ramprasathk07/LLM_voicechat.git
cd llm-voice-chatbot
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

3. Configure the settings in `config.py` (sample rate, TTS engine, etc.).

4. Run the application:
```bash
python app.py
```

## Usage
1. Start the chatbot by running `main.py`.
2. Speak into the microphone to interact with the chatbot.
3. The chatbot will process your speech and respond via text and voice.
4. Say "bye" to end the conversation.

## Code Structure
```plaintext
├── main.py                 # Main application
├── stt_whisper.py          # Speech-to-text module
├── tts.py                  # Text-to-speech module
├── LLM_engine.py           # LLM integration module
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Example Conversation
```plaintext
You: Hello, how are you?
Chatbot: I'm a voice-enabled chatbot here to assist you. How can I help today?
You: Can you tell me about Python?
Chatbot: Python is a versatile programming language popular for web development, data analysis, and more.
You: Bye.
Chatbot: Goodbye! Have a great day.
```

## Future Enhancements
- Add multilingual support for transcription and responses.
- Enhance response personalization using user profiles.
- Improve real-time processing with low-latency models.

## Contributions
Contributions are welcome! Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- [OpenAI Whisper](https://openai.com/whisper)
- [WebRTC VAD](https://webrtc.org/)
- [Hugging Face Transformers](https://huggingface.co/)
- [gTTs](https://gtts.readthedocs.io/en/latest/)

