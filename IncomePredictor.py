import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn import metrics
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

y_column_name = 'Income in EUR'
instance = None
selected_training_columns = ['Year of Record',
                             'Age',
                             'Gender',
                             'Country',
                             'Size of City',
                             'University Degree',
                             'Wears Glasses',
                             'Hair Color',
                             'Profession',
                             'Body Height [cm]',
                             y_column_name
                             ]


def read_data():
    labelled_dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
    unlabelled_dataset = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
    instance = unlabelled_dataset['Instance']
    return labelled_dataset, unlabelled_dataset, instance


def remove_outliers(dataset):
    dataset = dataset[dataset['Income in EUR'] > 0]
    return dataset


def preprocess_dataset(dataset):
    dataset = dataset[selected_training_columns]
    process_gender(dataset)
    process_age(dataset)
    process_year_of_record(dataset)
    process_profession(dataset)
    process_university_degree(dataset)
    process_hair_color(dataset)
    return dataset


def process_hair_color(dataset):
    dataset['Hair Color'].replace('0', 'Missing', inplace=True)
    dataset['Hair Color'].replace('Unknown', 'Missing', inplace=True)
    dataset['Hair Color'].replace(np.nan, 'Missing', inplace=True)


def process_university_degree(dataset):
    dataset['University Degree'].replace(np.nan, 'Missing', inplace=True)


def process_profession(dataset):
    dataset['Profession'].replace(np.nan, 'Missing', inplace=True)


def process_year_of_record(dataset):
    year_median = dataset['Year of Record'].median()
    dataset['Year of Record'].replace(np.nan, year_median, inplace=True)


def process_age(dataset):
    age_median = dataset['Age'].median()
    dataset['Age'].replace(np.nan, age_median, inplace=True)
    dataset['Age'] = (dataset['Age'] * dataset['Age']) ** (0.5)


def process_gender(dataset):
    dataset['Gender'].replace('0', 'Missing', inplace=True)
    dataset['Gender'].replace('Other', 'Missing', inplace=True)
    dataset['Gender'].replace('Unknown', 'Missing', inplace=True)
    dataset['Gender'].replace(np.nan, 'Missing', inplace=True)


def predict_prod_data(unlabelled_preprocessed_dataset, regressor):
    Y_prod_predictions = np.exp(regressor.predict(unlabelled_preprocessed_dataset))
    Y_prod_predictions = Y_prod_predictions
    df = pd.DataFrame({'Instance': instance, 'Income': Y_prod_predictions})
    df.to_csv('Submission.csv', index=False)


def encode(labelled_data, unlabelled_data, columns):
    encoder = OneHotCategoricalEncoder(
        top_categories=None,
        variables=columns,  # we can select which variables to encode
        drop_last=True)
    encoder.fit(labelled_data)
    labelled_data = encoder.transform(labelled_data)
    unlabelled_data = encoder.transform(unlabelled_data)
    return labelled_data, unlabelled_data

def calculate_metrics():
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#### 1. read data #######
labelled_dataset, unlabelled_dataset, instance = read_data()
######################

#### 2. remove outliers########
labelled_dataset = remove_outliers(labelled_dataset)
########################


##### 3. do preprocessing #########
labelled_preprocessed_dataset = preprocess_dataset(labelled_dataset)
unlabelled_preprocessed_dataset = preprocess_dataset(unlabelled_dataset)
#################################

######### 4. extract y column #########
y = labelled_preprocessed_dataset[y_column_name]
labelled_preprocessed_dataset.drop(y_column_name, axis=1, inplace=True)
unlabelled_preprocessed_dataset.drop(y_column_name, axis=1, inplace=True)
############################

######## 5. Do hot encoding #########
encoding_column_list = ['Gender', 'Country', 'Hair Color', 'University Degree', 'Profession']
labelled_preprocessed_dataset, unlabelled_preprocessed_dataset = encode(labelled_preprocessed_dataset,
                                                                        unlabelled_preprocessed_dataset,
                                                                        encoding_column_list)
#############################


####### 6. Training the model ############
X_train, X_test, y_train, y_test = train_test_split(labelled_preprocessed_dataset, y, test_size=0.2, random_state=0)
regressor = LinearRegression()

##### 7. transform y ###############
y_train_log = np.log(y_train)
###################################

###### 8. fit model and predict the income ###############
regressor.fit(X_train, y_train_log)
y_pred = np.exp(regressor.predict(X_test))
#################################################

######## 9. calculate metrics ######################
calculate_metrics()
###########################################

######## 10. Predict production data ################
predict_prod_data(unlabelled_preprocessed_dataset, regressor)
####################################################
