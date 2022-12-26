import flask
from flask import request,render_template
app = flask.Flask(__name__)
app.config["DEBUG"] = True
import pickle
from flask_cors import CORS
CORS(app)
model = pickle.load(open('model.pkl', 'rb'))
# main index page route
@app.route('/')
def home():
    return  render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    model=pickle.load(open('model.pkl','rb'))
    age = model.predict([[int(request.args['gender']),
                            int(request.args['religion']),
                            int(request.args['caste']),
                            int(request.args['mother_tongue']),
                            int(request.args['country']),
                            int(request.args['height_cms']),
                           ]])

    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text="Your Flight Price is Rs. {}".format(output))
      #return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)