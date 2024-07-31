from flask import Flask, request, render_template, url_for
from TTS.api import TTS
import torch
import os
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up Coqui TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def chunk_text(text, chunk_size=1500):
    paragraphs = text.split('\n\n')
    chunks = []
    for paragraph in paragraphs:
        while len(paragraph) > chunk_size:
            cut_off = paragraph.rfind(' ', 0, chunk_size)
            if cut_off == -1:
                cut_off = chunk_size
            chunks.append(paragraph[:cut_off])
            paragraph = paragraph[cut_off:].strip()
        if paragraph:
            chunks.append(paragraph)
    return chunks


def new_split_into_sentences(self, text):
    sentences = self.seg.segment(text)
    sentences_without_dots = []
    for sentence in sentences:
        if sentence.endswith('.') and not sentence.endswith('...'):
            sentence = sentence[:-1]

        sentences_without_dots.append(sentence)

    return sentences_without_dots


Synthesizer.split_into_sentences = new_split_into_sentences


@app.route('/tts', methods=['POST'])
def tts_synthesize():
    voice = request.form['voice']
    language = request.form['language']
    text = request.form.get('text', '')

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        text = file.read().decode('utf-8')

    chunks = chunk_text(text)
    combined_audio = AudioSegment.empty()

    for chunk in chunks:
        tts.tts_to_file(text=chunk, speaker_wav=voice, language=language, file_path="chunk.wav")
        chunk_audio = AudioSegment.from_wav("chunk.wav")
        combined_audio += chunk_audio

    # Save the combined audio to a file
    output_path = os.path.join('static', 'output.mp3')
    combined_audio.export(output_path, format="mp3")

    audio_url = url_for('static', filename='output.mp3')
    return render_template('index.html', audio_url=audio_url)


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0')
