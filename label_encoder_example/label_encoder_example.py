import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("datasets/play_tennis.csv")

label_encoder = LabelEncoder()
outlook_encoder = LabelEncoder()

data['PlayTennis'] = label_encoder.fit_transform(data['PlayTennis'])
data["Outlook"] = outlook_encoder.fit_transform(data["Outlook"])

print("After Encoding: ")
print(data)

print(label_encoder.classes_)
print(outlook_encoder.classes_)
