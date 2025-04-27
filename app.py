from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('disease_prediction_model.pkl')

# Labels & Mapping
labels = ['Mức Độ Sốt (F)', 'Mức Độ Đau Đầu (1-10)', 'Mức Độ Ho(1-10)', 'Mức Độ Mệt Mỏi(1-10)', 'Mức Độ Đau Nhức Cơ Thể(1-10)']
disease_dict = {
    'Runny Nose': 'Sổ mũi',
    'Cough': 'Ho',
    'Common Cold': 'Cảm lạnh thông thường',
    'Body Ache': 'Đau nhức cơ thể',
    'Malaria': 'Sốt rét',
    'Asthma': 'Hen suyễn',
    'Normal Fever': 'Sốt thông thường',
    'Dengue': 'Sốt xuất huyết'
}

@app.route('/')
def home():
    return render_template('index.html', labels=labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[label]) for label in labels]
        prediction = model.predict([data])[0]
        result = disease_dict.get(prediction)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
