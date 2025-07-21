Titanic Survival Prediction – EDA & Machine Learning
This project performs rigorous Exploratory Data Analysis (EDA) and builds a Logistic Regression classifier on the Titanic dataset. The aim is to explore passenger features, clean and transform the data, and build a baseline predictive model for survival classification.

Project Objectives
Perform in-depth data exploration and visualization.

Handle missing values with custom heuristics.

Engineer and transform features.

Train and evaluate a Logistic Regression model for binary classification.

Build a reproducible ML pipeline using standard Python libraries.

Technologies Used
Python 3.x

pandas

numpy

seaborn

matplotlib

scikit-learn

Workflow Summary
Data Loading and Initial Inspection

Missing Data Analysis via Heatmaps

Survival Relationship Exploration

Age Imputation by Passenger Class

Feature Engineering (e.g., one-hot encoding)

Model Training (Logistic Regression)

Prediction on Holdout/Test Set

1. Exploratory Data Analysis (EDA)
Missing Data Visualization
python
Copy
Edit
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
Survival Distribution
python
Copy
Edit
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', hue='Sex', data=train)
sns.countplot(x='Survived', hue='Pclass', data=train)
Age Distribution and Imputation
python
Copy
Edit
sns.boxplot(x='Pclass', y='Age', data=train)

def impute_age(cols):
    Age, Pclass = cols
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
2. Data Preprocessing
Dropping Sparse/Irrelevant Features
python
Copy
Edit
train.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
Categorical Encoding
python
Copy
Edit
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex', 'Embarked'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)
3. Machine Learning – Logistic Regression
Train/Test Split
python
Copy
Edit
from sklearn.model_selection import train_test_split

X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
Model Training
python
Copy
Edit
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
Prediction and Evaluation
python
Copy
Edit
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
Results
The Logistic Regression model was trained successfully and evaluated on a 30% holdout set.

Output includes precision, recall, f1-score, and accuracy metrics.

The code is written to be modular and reproducible for rapid experimentation.

How to Run
Clone this repository.

Install required Python libraries using pip install -r requirements.txt.

Open EDA2.ipynb in Jupyter Notebook or Google Colab.

Execute each cell sequentially to reproduce results.

License
This project is open-source under the MIT License.
