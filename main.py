from LLM_engine import LLM
from stt_whisper import STT
from tts import TTS

# Initialize the components
speech_to_text = STT()
llm_engine = LLM()
text_to_speech = TTS(audio_paths='Response')

print("Voice Chatbot is live! Say 'bye' to exit.")

while True:
    print("Listening...")
    txt = speech_to_text.stt() 
    print(f"You: {txt}")

    if txt == "ending":
        print("Chatbot: Goodbye! Have a great day!")
        text_to_speech.synthesize(txt="Goodbye! Have a great day!")
        break

    response = llm_engine.Groq_LLM(txt, word_limit=20)
    print(f"Chatbot: {response}")

    text_to_speech.synthesize(txt=response)




