import pandas as pd
from flask import Flask, jsonify, request
from sklearn import preprocessing
import pickle
import numpy as np

# load model
model = pickle.load(open('Models/dummyModel.pkl','rb'))
scaler = pickle.load(open('Models/scaler.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    df = pd.DataFrame.from_dict(data)

    #Feature Engineering
    #Replace Special Characters
    df['CommentClean'] = df['Comment'].str.replace('[^a-zA-Z]', ' ')
    df['CommentClean'] = df['CommentClean'].str.lower()
    df['CommentLength'] = df['CommentClean'].str.split().str.len()

    #Average Word Length
    df['CommentCharacters'] = df['CommentClean'].str.len()
    df['AvgWordLength'] = df['CommentCharacters'] / df['CommentLength']

    #Drop Columns
    df2 = df.drop(['Comment', 'CommentClean', 'Subreddit'], axis=1)

    #Instantiate MinMaxScaler
    df2 = scaler.transform(df2)

    #Fit and Transform X_train
    temp = np.array(df2).reshape((1, -1))

    
    # predictions
    result = model.predict(temp)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

