import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas.core.common import random_state
from plotly.offline import iplot, plot
from plotly.subplots import make_subplots
from matplotlib.colors import ListedColormap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings

warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

colors = ["#b1e7cd", "#854442", "#000000", "#fff4e6", "#3c2f2f",
          "#be9b7b", "#512E5F", "#45B39D", "#AAB7B8 ", "#20B2AA",
          "#FF69B4", "#00CED1", "#FF7F50", "#7FFF00", "#DA70D6"]

filePath = "../Datasets/DSSalary/ds_salaries.csv"
df = pd.read_csv(filePath)


# summarize the dataset
def summarize_dataset(data_frame):
    print(f"Rows: {data_frame.shape[0]}\nColumns: {data_frame.shape[1]}")
    print(data_frame.info())
    summary = pd.DataFrame({
        'count': data_frame.shape[0],
        'nulls': data_frame.isnull().sum(),
        'null(%)': data_frame.isnull().mean() * 100,
        'cardinality': data_frame.nunique()
    })
    print(summary)
    print(data_frame.describe(include=np.number))
    print(data_frame.describe(include=object))


def sum_of_salary_per_year(data_frame):
    sum_of_salary = data_frame.groupby('work_year')['salary'].sum()

    plt.plot(sum_of_salary.index, sum_of_salary.values, marker='o')
    plt.xticks(ticks=sum_of_salary.index)
    plt.yticks(ticks=sum_of_salary.values)
    plt.title("Sum of Salary Per Year")
    plt.xlabel("Work Year")
    plt.ylabel("Sum of Salary")
    plt.show()


def first_ten_job_count(data_frame):
    first_ten_jobs = data_frame['job_title'].value_counts()[:10]

    plt.scatter(
        x=first_ten_jobs.index,
        y=first_ten_jobs.values,
        s=[1.1 * val for val in first_ten_jobs.values],
        marker='o',
        alpha=0.4,
        c='b',
        vmin=1.0,
        vmax=10.0
    )
    plt.xticks(rotation=-60)
    plt.show()


def top_ten_job_salaries(dframe):
    top_ten = dframe.groupby("job_title")["salary"].sum().sort_values(ascending=False).head(10)

    plt.plot(
        top_ten.index,
        top_ten.values,
        marker='o'
    )
    plt.xticks(ticks=top_ten.index)
    plt.yticks(ticks=top_ten.values)
    plt.title("Top Ten Salaries")
    plt.xlabel("Job Titles")
    plt.ylabel("Salary")
    plt.show()
    

def avg_salary_by_company_location(dframe):
    means = dframe.groupby('company_location')['salary_in_usd'].mean().reset_index()

    fig = px.choropleth(
        means,
        locations='company_location',
        locationmode='USA-states',
        color='salary_in_usd',
        color_continuous_scale='Viridis',
        title='Average Salary by Company Location',
        labels={'salary_in_usd': 'Average Salary in USD'},
        template='plotly_dark'
    )

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor='Black',
        showland=True,
        landcolor='rgb(243, 243, 243)',
        showocean=True,
        oceancolor='rgb(204, 204, 255)',
        showlakes=True,
        lakecolor='rgb(127,205,255)',
    )
    fig.show()


def get_xy(dframe):
    x = dframe.drop(['company_size'], axis=1)
    y = dframe['company_size']
    return x, y

def encode_categorical_columns(x):
    for column in x.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        x[column] = le.fit_transform(x[column])
    return x


def encode_target(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return y

if __name__ == "__main__":
    x, y = get_xy(df)
    x = encode_categorical_columns(x)
    y = encode_target(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the features
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # Train model
    xgb = XGBClassifier(random_state=40)
    xgb.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = xgb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Report: \n{report}')

