import base64
import os
import youtube_dl
import moviepy.editor as moviepy

from dotenv import load_dotenv

from flask import Flask, jsonify, request

import inference


app = Flask(__name__)

load_dotenv()

@app.route('/get_api_key', methods=['POST'])
def GetApiKey():
    if request.method == 'POST':
        data = {
            "api_key": os.getenv(api_key)
        }

        return jsonify(data)


def check_valid_api_key(api_key):
    return api_key == os.getenv(api_key)


def convert_to_wav(param):
    pass


@app.route('/transcribe_file', methods=['POST'])
def TranscribeAudio():
    if request.method == 'POST' and check_valid_api_key(request.form.get('api_key')):
        if request.form.get('file_datatype') == 'a_upload':
            enc_file = request.files['file']
            enc_file.save(os.path.join('./uploads', 'live.wav'))
        elif request.form.get('file_datatype') == 'v_upload':
            convert_to_wav(request.files['file'])
        elif request.form.get('file_datatype') == 'y_video':
            if os.path.exists("./uploads/live.webm"):
                os.remove("./uploads/live.webm")
            if os.path.exists("./uploads/live.wav"):
                os.remove("./uploads/live.wav")
            ydl_opts = {'outtmpl': './uploads/live.webm'}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([request.form.get('file')])
            clip = moviepy.VideoFileClip("./uploads/live.mkv")
            clip.audio.write_audiofile("./uploads/live.wav")
        file = open('.env', 'w+')
        modelName = request.form.get('model_name')
        e = "device = cpu\nmodel = " + modelName
        file.write(e)
        file.close()
        result = inference.infer(modelName=modelName)
        data = {"transcript": result[0],
                "latency": result[1]}
        return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
