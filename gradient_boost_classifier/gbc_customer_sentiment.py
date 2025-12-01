import pandas as pd

# import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted")

# print(os.getcwd())

data = pd.read_csv("datasets/customer_sentiment.csv")

# sentiment distribution by gender with countplot
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.countplot(data=data, x="gender", hue="sentiment", ax=ax)
# ax.set_title("Sentiment Distribution by Gender", fontsize=16, fontweight="bold", pad=20)
# ax.set_xlabel("Gender", fontsize=12)
# ax.set_ylabel("Sentiment", fontsize=12)
# plt.savefig("gender_sentiment.png")

# sentiment distribution by age group
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.countplot(data=data, x="age_group", hue="sentiment", ax=ax)
# ax.set_title(
#     "Sentiment Distribution by Age Group", fontsize=16, fontweight="bold", pad=20
# )
# ax.set_xlabel("Age Group", fontsize=12)
# ax.set_ylabel("Sentiment", fontsize=12)

"""
From the countplot we see that sentimant is distributed more or less equally amonth all the genders.
So we can say that the feature gender has no influence on sentiment. But before we conclude we will
do the Chi-Square test to confirm. 
"""
feature_encoders = {}
# dropping review_text because label encoding text data as integers is statistically meaningless
# for text based feature we need to use methods like TF-IDF, bag of words, embeddings (BERT)
X = data.drop(["customer_id", "sentiment", "review_text"], axis=1)

for column in X.columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    feature_encoders[column] = encoder

Y = data["sentiment"]

X_csr = csr_matrix(X.values)

tf_idf = TfidfVectorizer(lowercase=True)
# review_text is a sparsed matrix
review_text = tf_idf.fit_transform(data["review_text"])

X_combined = hstack([X_csr, review_text])

feature_column_list = X.columns.tolist()
feature_column_list.append("review_text")

chi_scores, p_values = chi2(X_combined, Y)

feature_chi2_result = pd.DataFrame(
    {
        "features": feature_column_list,
        "chi_square_score": chi_scores,
        "p_values": p_values,
    }
)

feature_chi2_result = feature_chi2_result.sort_values(
    by="chi_square_score", ascending=False
)

print(feature_chi2_result)
# feature_chi2_result.to_csv("datasets/feature_chi2_result.csv")

# plt.tight_layout()
# plt.show()
# plt.close()
