import torch
import torch.nn as nn
import torchaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import os
from datetime import datetime

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class VoiceEncoder(nn.Module):
    """Neural network to capture voice characteristics"""
    def __init__(self):
        super(VoiceEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.lstm = nn.LSTM(256, 512, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024, 256)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

class ChatterBoxTTS:
    def __init__(self):
        self.sample_rate = 22050
        self.device = device
        
        # Create output directory
        self.output_dir = self.create_output_directory()
        
        # Initialize voice encoder
        self.voice_encoder = VoiceEncoder().to(self.device)
        
        # Load pretrained TTS model
        self.load_tts_model()
        
        print("ChatterBox TTS initialized!")
        print(f"Audio files will be saved to: {self.output_dir}")
    
    def create_output_directory(self):
        """Create a dedicated folder for audio files on Desktop"""
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_dir = os.path.join(desktop_path, "ChatterBox_Audio_Output")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def load_tts_model(self):
        """Load pretrained TTS model"""
        try:
            bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
            self.tacotron2 = bundle.get_tacotron2().to(self.device)
            self.vocoder = bundle.get_vocoder().to(self.device)
            self.processor = bundle.get_text_processor()
            print("TTS models loaded successfully!")
        except Exception as e:
            print(f"Could not load TTS models: {e}")
            self.tacotron2 = None
    
    def record_voice(self, duration=5):
        """Record voice from microphone"""
        print(f"Recording for {duration} seconds...")
        try:
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1,
                          dtype='float32')
            sd.wait()
            return audio.flatten()
        except Exception as e:
            print(f"Recording error: {e}")
            return np.array([])
    
    def extract_voice_print(self, audio):
        """Extract voice characteristics"""
        if len(audio) == 0:
            return np.zeros(256)
            
        try:
            if len(audio) > 16000:
                audio = audio[:16000]
            
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                voice_print = self.voice_encoder(audio_tensor)
            return voice_print.cpu().numpy()
        except:
            return np.zeros(256)
    
    def generate_speech(self, text, emotion='neutral'):
        """Generate speech from text"""
        if self.tacotron2 is None or len(text.strip()) == 0:
            return self.fallback_tts(text, emotion)
        
        try:
            with torch.no_grad():
                processed, lengths = self.processor(text)
                processed = processed.to(self.device)
                lengths = lengths.to(self.device)
                
                spec, spec_lengths, _ = self.tacotron2.infer(processed, lengths)
                waveforms, lengths = self.vocoder(spec, spec_lengths)
                
            audio = waveforms[0].cpu().numpy()
            audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.8
            
            audio = self.apply_emotion_to_speech(audio, emotion)
            
            return audio
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return self.fallback_tts(text, emotion)
    
    def apply_emotion_to_speech(self, audio, emotion):
        """Apply emotional effects to generated speech"""
        if len(audio) == 0:
            return audio
            
        audio = audio.copy()
        
        emotion_effects = {
            'happy': {'pitch_shift': 1.2, 'speed': 1.1},
            'sad': {'pitch_shift': 0.8, 'speed': 0.9},
            'angry': {'pitch_shift': 1.3, 'speed': 1.2},
            'excited': {'pitch_shift': 1.4, 'speed': 1.3},
            'calm': {'pitch_shift': 0.9, 'speed': 0.95},
            'neutral': {'pitch_shift': 1.0, 'speed': 1.0}
        }
        
        params = emotion_effects.get(emotion, emotion_effects['neutral'])
        
        try:
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, 
                                              n_steps=params['pitch_shift'] - 1.0)
            audio = librosa.effects.time_stretch(audio, rate=params['speed'])
        except Exception as e:
            print(f"Emotion effect error: {e}")
        
        return audio
    
    def fallback_tts(self, text, emotion):
        """Simple fallback TTS"""
        duration = max(1.0, len(text) * 0.15)
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        emotion_params = {
            'happy': {'freq': 220, 'mod': 0.1},
            'sad': {'freq': 160, 'mod': 0.05},
            'angry': {'freq': 190, 'mod': 0.15},
            'excited': {'freq': 250, 'mod': 0.2},
            'calm': {'freq': 180, 'mod': 0.08},
            'neutral': {'freq': 200, 'mod': 0.1}
        }
        
        params = emotion_params.get(emotion, emotion_params['neutral'])
        wave = 0.5 * np.sin(2 * np.pi * params['freq'] * t)
        wave += 0.3 * np.sin(2 * np.pi * params['freq'] * 2 * t)
        wave += params['mod'] * np.sin(2 * np.pi * 5 * t)
        
        return wave
    
    def clone_voice(self, reference_audio, text, emotion='neutral'):
        """Clone voice from reference audio"""
        voice_print = self.extract_voice_print(reference_audio)
        speech = self.generate_speech(text, emotion)
        return self.apply_voice_print(speech, voice_print)
    
    def apply_voice_print(self, audio, voice_print):
        """Apply voice characteristics to audio"""
        if len(audio) == 0:
            return audio
            
        audio = audio.copy()
        
        try:
            freqs = np.fft.rfft(audio)
            mod_factor = 1 + 0.1 * np.mean(voice_print) if np.any(voice_print) else 1.0
            freqs = freqs * mod_factor
            audio = np.fft.irfft(freqs)
        except:
            pass
            
        return audio
    
    def play_audio(self, audio):
        """Play audio through speakers"""
        if len(audio) > 0:
            try:
                sd.play(audio, self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")
        else:
            print("No audio to play")
    
    def save_audio(self, audio, filename):
        """Save audio to file with full path"""
        if len(audio) > 0:
            try:
                # Create full file path in the output directory
                full_path = os.path.join(self.output_dir, filename)
                sf.write(full_path, audio, self.sample_rate)
                print(f"✅ Audio saved as: {full_path}")
                return full_path
            except Exception as e:
                print(f"Save error: {e}")
                return None
        else:
            print(" No audio to save")
            return None
    
    def auto_save_audio(self, audio, prefix="auto_save"):
        """Automatically save audio with timestamp"""
        if len(audio) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.wav"
            return self.save_audio(audio, filename)
        return None

def main():
    """Main interactive function"""
    tts = ChatterBoxTTS()
    
    print("🎤 Welcome to ChatterBox TTS with Voice Cloning!")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Generate speech with emotion")
        print("2. Voice cloning demo") 
        print("3. Auto-save test (no prompts)")
        print("4. Open output folder")
        print("5. Exit")
        
        try:
            choice = input("\nChoose an option (1-5): ").strip()
        except:
            break
        
        if choice == '1':
            text = input("Enter text to speak: ").strip()
            if not text:
                print("Please enter some text!")
                continue
                
            emotion = input("Enter emotion (neutral/happy/sad/angry/excited/calm): ").strip().lower()
            emotions = ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm']
            if emotion not in emotions:
                emotion = 'neutral'
                print("Using neutral emotion")
            
            print("Generating speech...")
            speech = tts.generate_speech(text, emotion)
            tts.play_audio(speech)
            
            # AUTO-SAVE - No prompt needed
            saved_path = tts.save_audio(speech, f"chatterbox_{emotion}.wav")
            if saved_path:
                print(f"🎵 Audio automatically saved!")
                
        elif choice == '2':
            print("First, record reference voice (3 seconds):")
            ref_audio = tts.record_voice(duration=3)
            
            if len(ref_audio) > 0:
                text = input("Enter text for voice cloning: ").strip()
                if not text:
                    text = "Hello, this is my cloned voice"
                    
                emotion = input("Enter emotion: ").strip().lower()
                emotions = ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm']
                if emotion not in emotions:
                    emotion = 'neutral'
                
                print("Cloning voice and generating speech...")
                cloned_speech = tts.clone_voice(ref_audio, text, emotion)
                tts.play_audio(cloned_speech)
                
                # AUTO-SAVE - No prompt needed
                saved_path = tts.save_audio(cloned_speech, f"cloned_voice_{emotion}.wav")
                if saved_path:
                    print(f" Cloned audio automatically saved!")
            else:
                print("Recording failed!")
                
        elif choice == '3':
            # Test auto-save without any prompts
            test_text = "This is a test of automatic audio saving feature."
            print(f"Auto-generating: '{test_text}'")
            speech = tts.generate_speech(test_text, 'happy')
            tts.play_audio(speech)
            saved_path = tts.auto_save_audio(speech, "test_audio")
            if saved_path:
                print("Test audio auto-saved successfully!")
                
        elif choice == '4':
            # Open the output folder
            try:
                os.system(f'open "{tts.output_dir}"')
                print(f"Opened output folder: {tts.output_dir}")
            except Exception as e:
                print(f"Could not open folder: {e}")
                
        elif choice == '5':
            print("Goodbye! ")
            break
            
        else:
            print("Invalid choice! Please choose 1-5.")

if __name__ == "__main__":
    main()