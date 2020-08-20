"""
An employee attrition data to predict the likelyhood of employee retention.
The following points should be noted:
(1)  Since the data splits the employees into two: those who have left and existing employees,
It is necessary to first add an 'Attrition' column and then bring them together as one solid dataset.
This is done using the '.append()' method.
(2)  There's need to visualize the data being described. Hence a chart is made and summary statistics.
(3)  Categorical columns from the data are encoded

The Accuracy of the Trainng Model is about 99.8%

The confusion Matrix and Accuracy Score for the test data is about 99.2%

"""


####Importin the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#### Adding the 'Attrition' Column Joining the datasets
d1 = pd.read_excel("C:/Users/mowab/Downloads/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",
                   sheet_name='Existing employees')
d1['Attrition'] = 'No'
print(d1)

d2 = pd.read_excel("C:/Users/mowab/Downloads/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",
                   sheet_name='Employees who have left')
d2['Attrition'] = 'Yes'
print(d2)


#### Joining the datasets Together
dataset = d1.append(d2)
print(dataset)


####Visualizing the Attrition data
sns.countplot(dataset['Attrition'])
plt.show()


# fig_dimensions = (120, 100)
# fig, ax = plt.subplot(figsize = fig_dimensions)
# sns.countplot(x = 'dept', hue = 'Attrition', data=dataset, palette='colorblind', ax=ax, edgecolor= sns.color_palette('dark', n_colors=1))
# plt.show()


##### Visualizing a summary of the columns
for column in dataset.columns:
    if dataset[column].dtype == object:
        print(str(column) + ' : ' + str(dataset[column].unique()))
        print(dataset[column].value_counts())
        print("\n ****************************************************** \n")
    elif dataset[column].dtype == np.number:
        print(dataset[column].value_counts())
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


##### Dropping the 'Employee ID' column
dataset = dataset.drop('Emp ID', axis=1)


#### Visualizing the correlation between the columns
print("The Correlation between the columns are:   \n")
print(dataset.corr())


##### Visualizing the Heatmap of the correlation
plt.figure(figsize=(9, 9))
sns.heatmap(dataset.corr(), annot=True, fmt='.0%')
plt.show()


#### Defining the dependent and Independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 9].values
print(y)


#### Importing preprocessing Libraries for encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

col_tran = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [7, 8])],
                             remainder='passthrough')

##### Encoding the independent variable and Dependent variabes
x = np.array(col_tran.fit_transform(x), dtype=np.float)

lab_en = LabelEncoder()
y = lab_en.fit_transform(y)

print(x)
print(y)


#### Spliting the data into training and Testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)


##### Fitting the training set with the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
forest.fit(x_train, y_train)

#### Calculaing the Score
score = forest.score(x_train, y_train)
print(score)


"""
Showing the confusion matrix and accuracy for  the model on the test data
Classification accuracy is the ratio of correct predictions to total predictions made.
"""

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, forest.predict(x_test))

aa = cm[0][0]
ab = cm[0][1]
ba = cm[1][0]
bb = cm[1][1]

cm1 = ((aa + bb) / (aa + ab + ba + bb)) * 100
print('\nThe Accuracy of the Model is :  {}%!\n'.format(cm1))

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test, forest.predict(x_test))
print('\nThe Accuracy Score for the Model is :  {}% '.format(acc_score * 100))
