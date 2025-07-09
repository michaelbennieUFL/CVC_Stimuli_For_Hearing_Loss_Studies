"""
phoneme_tts.py – ElevenLabs direct ARPAbet phoneme-to-speech wrapper
pip install elevenlabs python-dotenv
"""

import os
from typing import Optional

from elevenlabs import ElevenLabs, play, VoiceSettings, PronunciationDictionaryVersionLocator  # type: ignore
from elevenlabs.pronunciation_dictionaries import PronunciationDictionaryRule_Phoneme


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

                speed=1.0,

            ),

        )

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id

    def speak_phonemes(
            self,
            phoneme_str: str,
            output_path: Optional[str] = None,
    ) -> bytes:
        """
        Takes an ARPAbet phoneme string (e.g. "Z UW N") and converts it to speech.

        Returns raw WAV audio bytes. Plays the audio and optionally writes it to disk.
        """
        ssml_text = f'<phoneme alphabet="cmu-arpabet" ph="{phoneme_str}"></phoneme>'

        # get stream
        audio_stream = self.client.text_to_speech.convert(
            text=ssml_text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_192",
        )

        # join stream into bytes
        audio_bytes = b"".join(audio_stream)

        play(audio_bytes)

        if output_path:
            with open(output_path, "wb") as fp:
                fp.write(audio_bytes)

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

    phonemes = "B AH G"  # this should say "Zoon"

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
