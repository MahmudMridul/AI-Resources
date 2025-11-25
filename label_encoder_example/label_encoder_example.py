import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Datasets/play_tennis.csv")

encoder = LabelEncoder()

print("Before Encoding: ")
print(data)

data['PlayTennis'] = encoder.fit_transform(data['PlayTennis'])

print("After Encoding: ")
print(data)

print(encoder.classes_)
