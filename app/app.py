import os
from flask import Flask, request, render_template, jsonify
from pyngrok import ngrok
from inference import InferenceEnsemble

app = Flask(__name__)

# Config: Use Environment Variables for paths in production
BASE_DIR = os.environ.get("MODEL_DIR", "/content/drive/MyDrive") 
CONFORMER_PATH = os.path.join(BASE_DIR, "wav2vecconformer-finetune-checkpoints")
WAV2VEC2_PATH = os.path.join(BASE_DIR, "wav2vec2-finetune-checkpoints22/checkpoint-20000")
CUSTOM_PATH = os.path.join(BASE_DIR, "my_final_wav2vec2_model")
FUSION_PATH = os.path.join(BASE_DIR, "trained_fusion_model.pth")
NGROK_TOKEN = os.environ.get("NGROK_TOKEN")

system = None

def get_model():
    global system
    if system is None:
        try:
            system = InferenceEnsemble(CONFORMER_PATH, WAV2VEC2_PATH, CUSTOM_PATH, FUSION_PATH)
        except Exception as e:
            print(f"Failed to load models: {e}")
    return system

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = get_model()
    if not model:
        return jsonify({'error': 'Model failed to load'}), 500
        
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_path = "temp.wav"
    file.save(temp_path)
    
    try:
        text = model.predict(temp_path)
        return jsonify({'Final Ensemble': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        public_url = ngrok.connect(5000)
        print(f" * ngrok tunnel: {public_url}")
    app.run(port=5000)