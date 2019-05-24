import pandas as pd
from tqdm import tqdm
import ast
tqdm.pandas()

df = pd.read_csv('data/train.csv')


def dict_convert(data, key):
    fval = ''
    if data != 0:
        data = ast.literal_eval(data)
        # data = data[0]
        # data = data[key]
        for d in data:
            for x,y in d.items():
                if x == key:
                    fval += ' ' + y
                else:
                    pass
    return fval


df.drop(['id', 'homepage', 'imdb_id', 'poster_path'], axis=1, inplace=True)
# df['belongs_to_collection'].fillna(0, inplace=True)
# df['belongs_to_collection'] = df.progress_apply(lambda x: dict_convert(x['belongs_to_collection'], "name"), axis=1)
# df['genres'].fillna(0, inplace=True)
# df['genres'] = df.progress_apply(lambda x: dict_convert(x['genres'], "name"), axis=1)

print(df.dtypes)
print(df)