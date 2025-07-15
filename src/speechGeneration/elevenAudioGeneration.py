"""
phoneme_tts.py – ElevenLabs direct ARPAbet phoneme-to-speech wrapper
pip install elevenlabs python-dotenv
"""

import os
from typing import Optional

from elevenlabs import ElevenLabs, play, VoiceSettings, PronunciationDictionaryVersionLocator  # type: ignore
from elevenlabs.pronunciation_dictionaries import PronunciationDictionaryRule_Phoneme
from pydub import AudioSegment
from io import BytesIO
from pydub.playback import play as pydub_play
import librosa
import soundfile as sf

def speed_up_audio(audio_bytes: bytes, speed_factor: float = 1.3) -> bytes:
    """
    Tempo-scale `audio_bytes` by `speed` (e.g. 1.3 = +30 %) **without** altering pitch.
    Returns MP3 bytes.
    """
    # ── load bytes to float32 NumPy ───────────────────────────────────────────
    data, sr = sf.read(BytesIO(audio_bytes), dtype='float32')

    # Librosa expects (n_samples,) mono or (n_channels, n_samples) stereo
    if data.ndim == 1:
        y_out = librosa.effects.time_stretch(data, rate=speed_factor)
    else:
        # stereo → operate channel-wise
        y_out = librosa.effects.time_stretch(data.T, rate=speed_factor).T

    # ── back to bytes (MP3) ──────────────────────────────────────────────────
    out_buf = BytesIO()
    sf.write(out_buf, y_out, sr, format='wav')          # write WAV first
    out_buf.seek(0)

    # use pydub/ffmpeg to encode MP3 (192 kbps)
    mp3 = AudioSegment.from_file(out_buf, format='wav')
    mp3_buf = BytesIO()
    mp3.export(mp3_buf, format='mp3', bitrate='192k')
    return mp3_buf.getvalue()

class PhonemeTTSEngine:
    """
    Tiny ElevenLabs TTS wrapper for directly converting ARPAbet phonemes into speech.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_turbo_v2",
    ) -> None:
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id

        self.client.voices.settings.update(

            voice_id=self.voice_id,

            request=VoiceSettings(

                stability=1.0,

                use_speaker_boost=False,

                similarity_boost=0.0,

                style=0.0,

                speed=0.7,

            ),

        )

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id

    def speak_phonemes(
            self,
            phoneme_str: str,
            output_path: Optional[str] = None,
            speed_factor: float = 1.0,
    ) -> bytes:
        """
        Takes an ARPAbet phoneme string (e.g. "Z UW N") and converts it to speech.

        Returns raw WAV audio bytes. Plays the audio and optionally writes it to disk.
        """
        ssml_text = f'<phoneme alphabet="cmu-arpabet" ph="{phoneme_str}"></phoneme>.'

        # get stream
        audio_stream = self.client.text_to_speech.convert(
            text=ssml_text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_192",
        )

        # join stream into bytes
        audio_bytes = b"".join(audio_stream)



        if output_path:
            with open(output_path, "wb") as fp:
                fp.write(audio_bytes)

        if speed_factor != 1.0:
            audio_bytes = speed_up_audio(audio_bytes, speed_factor)

        play(audio_bytes)

        return audio_bytes

    def speak_with_dictionary(
        self,
        phoneme_str: str,
        grapheme: str = "qazxsw",
        alphabet: str = "cmu-arpabet",
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Convert `phoneme_str` to speech by creating a temporary pronunciation
        dictionary that maps `grapheme` (default: 'qazxsw') to the supplied
        phonemes.

        Returns raw WAV bytes, plays the audio, and optionally writes it to disk.
        """

        # ── 1. Build a one-rule dictionary on the fly ───────────────────────────
        rule = PronunciationDictionaryRule_Phoneme(
            string_to_replace=grapheme,
            phoneme=phoneme_str,
            alphabet=alphabet,
        )

        dictionary = self.client.pronunciation_dictionaries.create_from_rules(
            rules=[rule],
            name=f"tmp_{grapheme}",
        )

        # ── 2. Tell TTS to apply that specific dictionary version ───────────────
        locator = PronunciationDictionaryVersionLocator(
            pronunciation_dictionary_id=dictionary.id,
            version_id=dictionary.version_id,
        )

        # ── 3. Speak the placeholder word; the dictionary handles pronunciation ─
        audio_stream = self.client.text_to_speech.convert(
            text=grapheme,
            voice_id=self.voice_id,
            model_id=self.model_id,
            pronunciation_dictionary_locators=[locator],

            output_format="mp3_44100_192",
        )

        audio_bytes = b"".join(audio_stream)
        play(audio_bytes)

        if output_path:
            with open(output_path, "wb") as fp:
                fp.write(audio_bytes)

        return audio_bytes


# ───────────────────────────── demo usage ──────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())  # loads ELEVENLABS_API_KEY from .env

    tts = PhonemeTTSEngine(api_key=os.environ["ELEVENLABS_API_KEY"], voice_id=os.environ["ELEVENLABS_VOICE_ID"])

    phonemes = "B EH G"  # this should say "Zoon"

    wav = tts.speak_phonemes(
        phoneme_str=phonemes,
        output_path="output.wav",
    )


    # says “Zoon”
    tts.speak_with_dictionary("Y AO N")
    tts.speak_with_dictionary("M AO N")
    tts.speak_with_dictionary("N AO N")
    tts.speak_with_dictionary("R AO N")

    print("WAV written to output.wav – size:", len(wav), "bytes")
