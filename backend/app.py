from flask import Flask, render_template, request
import pickle
import numpy as np

# initiatialize flask app
app = Flask(__name__)

# load the model
model = pickle.load(open('bcmodel.pkl', 'rb'))

# home page
@app.route('/')
def defaultpage():
    return render_template('index.html')

# predict
@app.route('/', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    pred = model.predict(final_features)
    return render_template('index.html', results="Breast Cancer group is {}".format(pred))


if __name__ == "__main__":
    app.run(debug=True)