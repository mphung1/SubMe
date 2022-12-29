import base64
import os
import youtube_dl
import moviepy.editor as moviepy
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import inference
import random
from fpdf import FPDF
from google.cloud import storage

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}})

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-key.json'
storage_client = storage.Client()
bucket = storage_client.get_bucket('subme-transcription')

@app.route("/")
def hello_world():
    return "<p>Hello world! Welcome to SubMe API.</p>"

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
@cross_origin(origin='*',headers=['Content- Type','Authorization'])
def TranscribeAudio():
    if request.method == 'POST':
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
    else:
        return jsonify({"message":"check api key"})

@app.route('/export_file', methods=['POST'])
@cross_origin(origin='*',headers=['Content- Type','Authorization'])
def ExportFile():
    try:
        transcription_text = request.form.get('text');
        file_code = random.randint(0, 1000000000)
        file_path = "generatedFiles/" + str(file_code) + ".pdf"

        file = FPDF()
        file.add_page()
        file.set_font("Arial", size=12)
        file.cell(200, 10, txt="Transcript", ln=1, align="C")
        file.cell(200, 10, txt=transcription_text, ln=2, align="C")

        file.output(file_path)

        blob = bucket.blob(str(file_code)+".pdf")
        blob.upload_from_filename(file_path)

        if os.path.exists(file_path):
            os.remove(file_path)

        newResponse = {
          "fileLink": "https://storage.googleapis.com/subme-transcription/" + str(file_code) + ".pdf"
        }

        return jsonify(newResponse)

    except Exception as e:
        print(e)
        return jsonify({"message":"cannot export file"})

if __name__ == '__main__':
    app.run(debug=True)
