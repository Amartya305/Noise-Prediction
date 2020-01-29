from flask import Flask,request
from flask_restful import Resource,Api
import pickle
import pandas as pd
app = Flask(__name__)
api = Api(app)
class NoiseValue(Resource):
    def __init__(self):
        self.model = pickle.load(open('/models/multiple_regressor.pkl','rb'))
    def post(self):
        coords = request.get_json()
        coords.update((x,[y]) for x,y in coords.items())
        noiseVal = self.model.predict(pd.DataFrame.from_dict(coords))
        response = { 'lat': coords['lat'][0], 'long' : coords['long'][0], 'noiseVal' : noiseVal[0]}
        return response,201

api.add_resource(NoiseValue,'/predict')
if __name__ == '__main__':
    app.run(port=3000,debug=True)
