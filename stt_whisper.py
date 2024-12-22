import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from datetime import datetime
import os 
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
import webrtcvad
import pyaudio

class STT:
    def __init__(self):
        print("setting up STT module")
        self.audio_paths = f"saved_audios"
        os.makedirs(self.audio_paths,exist_ok=True)
        time = datetime.now()
        self.current_time = time.strftime("%Y-%m-%d_%H_%M")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-small"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)
        self.vad = webrtcvad.Vad()
        processor = AutoProcessor.from_pretrained(model_id)
        self.sample_rate = 16000
        self.frame_duration = 1000

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def record_audio(self,duration, sample_rate=16000, channels=1):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)
        # print("Recording...")
        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            frames.append(stream.read(1024))
        stream.stop_stream()
        stream.close()
        p.terminate()
        return b''.join(frames)

    def ensure_format(self,audio_data, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            raise ValueError(f"Unsupported sample rate: {sample_rate}. Expected {target_sample_rate}.")
        if len(audio_data.shape) > 1: 
            audio_data = audio_data[:, 0]
        audio_data = (audio_data * 32767).astype(np.int16)  
        return audio_data.tobytes()

    def stt(self):
        i = 0
        print("Starting real-time speech-to-text...")
        txt = ''
        frame_duration_ms = 20
        frame_size = int(self.sample_rate * frame_duration_ms / 1000) 

        while True:
            audio_data = self.record_audio(duration=3)

            audio_array = np.frombuffer(audio_data, dtype=np.int16)  
            audio_data = audio_array[:frame_size] 
            audio_bytes = audio_data.tobytes()  
            try:
                if not self.vad.is_speech(audio_bytes, self.sample_rate):
                    i+=1
                    if i>2:
                        i = 0
                        print("No speech detected, exiting...")
                        break

            except webrtcvad.Error as e:
                print(f"VAD processing error: {e}")
                continue
            
            normalized_audio = audio_array.astype(np.float32) / 32767.0
            result = self.pipe(normalized_audio)
            text = result.get("text", "").strip()
            
            a = txt.split()[-3:]
            
            a.append(text)
            if len(a)>3:
                if len(set(a)) <3:
                    break
            
            txt += text + " "
            print(f"TEXT:{txt}")
            if 'bye' in [i.lower() for i in text.split()] or 'bye.' in [i.lower() for i in text.split()]:
                    print("BYE")
                    return "ending"
            
        return txt

if __name__ == "__main__":
    obj = STT()
    obj.stt()