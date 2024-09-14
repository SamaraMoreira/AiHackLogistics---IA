from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/keras', methods=['POST'])
def uploadKeras():
    file = request.files['imagem']

    # Caminho para o modelo e rótulos
    model_path = r"C:\Users\samar_ca5ad0g\Desktop\Sprint 3 - IA\DataSet-ia\modelos\keras_model.h5"
    labels_path = r"C:\Users\samar_ca5ad0g\Desktop\Sprint3\models\labels.txt"

    # Carregar o modelo
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        return jsonify({'error': f'Error loading model: {str(e)}'}), 500

    # Carregar os rótulos
    try:
        with open(labels_path, "r") as f:
            class_names = f.readlines()
        class_names = [name.strip() for name in class_names]  
    except Exception as e:
        return jsonify({'error': f'Error loading labels: {str(e)}'}), 500

    # Criar o array da forma certa para alimentar o modelo Keras
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Processar a imagem
    image = Image.open(file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Converter para numpy array
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Fazer a previsão
    prediction = model.predict(data)
    print("Prediction Array:", prediction)  # Adicionar esta linha para verificar os valores brutos da previsão

    # Obter o índice da classe com a maior probabilidade
    if prediction.shape[1] == len(class_names):  # Verificar se o número de saídas do modelo corresponde ao número de classes
        index = np.argmax(prediction[0])
        class_name = class_names[index] if index < len(class_names) else "Unknown"
        confidence_score = prediction[0][index]
    else:
        class_name = "Unknown"
        confidence_score = 0.0

    # Imprimir a previsão e a pontuação de confiança
    print("Class:", class_name)
    print("Confidence Score:", confidence_score)

    data_atual = datetime.datetime.now()
    horario_atual = data_atual.strftime('%Y-%m-%d %H:%M:%S')

    return jsonify({
        'data': f'{horario_atual}',
        'class': f'{class_name}',
        'confidence': f'{confidence_score:.2f}',
        'prediction_array': prediction[0].tolist()  # Ajuste para retornar o array de previsão correto
    })

if __name__ == '__main__':
    app.run(debug=True)
