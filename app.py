import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
from flask import Flask,request,render_template
import numpy

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

le = LabelEncoder()

@app.route('/predict', methods = ['POST','GET'])
def predict():
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bp = int(request.form['bp'])
    chol = int(request.form['chol'])
    n_t_k = float(request.form['n_t_k'])

    features = (age,sex,bp,chol,n_t_k)
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    if prediction[0] == 1:
        res = 'drugA'
    elif prediction[0] == 2:
        res = 'drugB'
    elif prediction[0] == 3:
        res = 'drugC'
    elif prediction[0] == 4:
        res = 'drugX'
    elif prediction[0] == 5:
        res = 'DrugY'
    return render_template('index.html', prediction_text =f"Recommended drug is {res}")

if __name__ == '__main__':
    app.run(debug=True)