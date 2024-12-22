import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd
import warnings
import logging
import uuid
from datetime import datetime
import os 
from gtts import gTTS
import io
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("parler_tts").setLevel(logging.ERROR)

class TTS: 
    def __init__(self,audio_paths):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.audio_paths = audio_paths
        os.makedirs(self.audio_paths,exist_ok=True)

        time = datetime.now()
        self.current_time = time.strftime("%Y-%m-%d_%H_%M")
        # self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        
    
    def play_audio(self,audio_file):
        """Play back the recorded audio."""
        print("Playing back the recorded audio...")
        data, samplerate = sf.read(audio_file)  # Read with soundfile
        sd.play(data, samplerate)
        sd.wait()  # Wait until playback is complete

    def synthesize(self,txt="Hey, how are you doing today?"):
        print("Synthesizing our Audio...")
        description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
        
        self.filename = f"{self.audio_paths}/{uuid.uuid4().hex}_{self.current_time}.wav"
        # input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        # prompt_input_ids = self.tokenizer(txt, return_tensors="pt").input_ids.to(self.device)

        # generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        # audio_arr = generation.cpu().numpy().squeeze()

        tts = gTTS(text=txt, lang='en')
        tts.save(self.filename)
        
        # sf.write(self.filename, audio_arr, self.model.config.sampling_rate)
        self.play_audio(self.filename)
        # return audio_arr

if __name__ == "__main__":
    TTS()