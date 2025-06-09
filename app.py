from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your model
model = load_model('model.h5')

class_names = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte','Neutrophil']

cell_diseases = {
    'Basophil': [
        'Chronic Myeloid Leukemia (CML)',
        'Hypothyroidism',
        'Polycythemia vera'
    ],
    'Eosinophil': [
        'Asthma',
        'Eosinophilic Esophagitis',
        'Parasitic Infections'
    ],
    'Lymphocyte': [
        'Chronic Lymphocytic Leukemia (CLL)',
        'Infectious Mononucleosis',
        'Autoimmune Disorders'
    ],
    'Monocyte': [
        'Tuberculosis',
        'Monocytic Leukemia',
        'Chronic Infections'
    ],
    'Neutrophil': [
        'Bacterial Infections',
        'Acute Inflammation',
        'Neutrophilia'
    ]
}

# Example prediction function – customize as needed
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    label = class_names[class_id]

    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)

            if confidence < 0.50:  # You can adjust this
                error_message = "The uploaded image does not appear to be a valid white blood cell. Please upload a relevant image."
                return render_template('index.html', filename=file.filename, error=error_message)
            
            result_text = f"Predicted Class: {label} ({confidence * 100:.2f}%)"
            diseases = cell_diseases.get(label, [])
            return render_template('index.html', filename=file.filename, result=result_text, diseases=diseases)
    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return f"/static/uploads/{filename}"

if __name__ == '__main__':
    app.run(debug=True)
