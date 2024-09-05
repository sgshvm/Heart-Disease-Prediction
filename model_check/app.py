import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("check.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if "submit" in request.form:
        a = request.form.get('a',0)
        b = request.form.get('b',0)
        c = request.form.get('c',0)
        d = request.form.get('d',0)
        e = request.form.get('e',0)
        f = request.form.get('f',0)
        g = request.form.get('g',0)
        h = request.form.get('h',0)
        i = request.form.get('i',0)
        j = request.form.get('j',0)
        k = request.form.get('k',0)
        l = request.form.get('l',0)
        m = request.form.get('m',0)

        feature = [[a,b,c,d,e,f,g,h,i,j,k,l,m]]
        f = pd.DataFrame(feature,columns=['a','b','c','d','e','f','g','h','i','j','k','l','m'])

        f['a'] = f['a'].astype(float)
        f['b'] = f['b'].astype(float)
        f['c'] = f['c'].astype(float)
        f['d'] = f['d'].astype(float)
        f['e'] = f['e'].astype(float)
        f['f'] = f['f'].astype(float)
        f['g'] = f['g'].astype(float)
        f['h'] = f['h'].astype(float)
        f['i'] = f['i'].astype(float)
        f['j'] = f['j'].astype(float)
        f['k'] = f['k'].astype(float)
        f['l'] = f['l'].astype(float)
        f['m'] = f['m'].astype(float)

        data = np.asarray(f)
        input_data_reshaped = data.reshape(1,-1)


        prediction = model.predict(input_data_reshaped)

        if (prediction == 0):
            val = 'Person does not have heart disease'
        else:
            val = 'Person have heart disease' 

        return render_template("index.html", prediction_text = "prediction : {}".format(val))

if __name__ == "__main__":
    app.run(debug = True)