import dill
import pandas as pd
import os
import json

from datetime import datetime


def predict():

    with open(f'../data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl', 'rb') as file:
        model = dill.load(file)

    path_to_json = '../data/test'
    json_files = os.listdir(path_to_json)

    df = pd.DataFrame(columns=['namefile', 'pred'])

    for i in json_files:
        with open(f'{path_to_json}/{i}', 'r') as f:
            text = json.load(f)
            df1 = pd.DataFrame.from_dict([text])
            y = model.predict(df1)

            df = df.append({'namefile': i, 'pred': y}, ignore_index=True)

    df.to_csv(f'../data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}')


if __name__ == '__main__':
    predict()
