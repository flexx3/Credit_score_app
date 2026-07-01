#Import the necessary libraries to load data
import os
import pandas as pd
import sqlite3
from data import load_data
#import libraries for preprocessing and model building
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#libraries for visualization
import matplotlib.pyplot as plt
#libraries to save and load model
import joblib
from glob import glob
from pathlib import Path

class model_data:
    def __init__(self):
        self.None__ = None

    def wrangle(self, filepath):
        connection= sqlite3.connect(database='Insurance_data.db', check_same_thread=False)
        table_name= 'credit_card_default'
        data= load_data(connection=connection)
        #load csv data
        csv_data= data._load_csv(filepath)
        #insert csv data to db
        data._insert_into_db(table_name=table_name)
        db_data= data.read_data_from_db(table_name=table_name)
        return db_data
    

    #get features and target, split data into training and test sets
    def split_data(self):
        filepath = 'credit_card_default.csv'
        data = self.wrangle(filepath)
        #split data into  'features' and 'target'
        X = data.drop('default_payment_next_month', axis=1)
        y = data['default_payment_next_month']
        #check margin of class imbalance of target variable if greater than given threshold, balance the weigth of the model
        threshold = 25
        class_ = ''
        if (round((y.value_counts()[0]/len(y)) * 100) - round((y.value_counts()[1]/len(y)) * 100)) > (threshold):
            class_ = 'imbalanced'
        else:
            class_ = 'balanced'
        #split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        return class_, X_train, X_test, y_train, y_test

class model(model_data):
    def __init__(self):
        self.None_ = None
        self.model_directory = 'credit_model'
        self.model_subdirectory = 'trees'
        self.class_ = self.split_data()[0] #check if data is imbalanced, stored the result in a variable
        self.X_train = self.split_data()[1] #stores train data in a variable
        self.X_test = self.split_data()[2] # stores test data in a variable
        self.y_train = self.split_data()[3] #stores target train data in a variable
        self.y_test = self.split_data()[4]  #stores target test data in a variable


    #build model
    def build_model(self):
        X_train = self.X_train
        # Create numerical pipeline
        num_features_list= X_train.select_dtypes(include='number').columns.to_list()
        num_pipeline = Pipeline(steps=[
            ('simpleimputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        
        ])
        #create categorical pipeline
        cat_features_list= X_train.select_dtypes(include='category').columns.to_list()
        cat_list= [list(X_train[columns].dropna().unique()) for column in cat_features_list]
        
        cat_pipeline = Pipeline(steps=[
            ('simpleimputer', SimpleImputer(strategy='most frequent')),
             ('onehot', OneHotEncoder(categories=cat_list, handle_unknown='error', drop='first', sparse_output=False))
        ])
        #Transform individual cols
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features_list),
            ('cat', cat_pipeline, cat_features_list)
        ])
        class_weight = ''
        if self.class_ == 'imbalanced':
            class_weight = 'balanced'
        else:
            class_weight = None
        #create the model pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42, class_weight=class_weight))
        ])
        return pipeline

    #make predictions
    def get_predictions(self, feature):
        X_train = self.X_train
        #fit model to training data
        model_pipeline = self.build_model()
        model_pipeline.fit(X_train, self.y_train)
        predictions = model_pipeline.predict(feature)
        return predictions

    #get probabilities
    def get_probabilities(self):
        X_test = self.X_test
        #fit model to training data
        model_pipeline = self.build_model()
        model_pipeline.fit(self.X_train, self.y_train)
        #get the probabilities
        probabilities = model_pipeline.predict_proba(X_test)[:,1]
        return probabilities

    #get ROC scores
    def get_roc_score(self):
        y_test = self.y_test
        predictions = self.get_predictions()
        roc_auc = roc_auc_score(y_test, predictions)
        return roc_auc

    #get classification report
    def get_classification_report(self):
        y_test = self.y_test
        predictions = self.get_predictions()
        report = classification_report(y_test, predictions)
        return report

    #get confusion matrix
    def get_confusion_matrix(self):
        y_test = self.y_test
        predictions = self.get_predictions()
        matrix = confusion_matrix(y_test, predictions)
        return matrix

    #get auc-roc curve
    def get_auc_roc_curve(self):
        y_test = self.y_test
        #get auc-roc score
        auc_roc_score = self.get_roc_score()
        #get predicted probabilities
        probabilities = self.get_probabilities()
        fpr, tpr, thresholds = roc_curve(self.y_test, probabilities)
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % auc_roc_score)
        plt.plot([0,1], [0,1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc='lower right')
        return plt.show()

    #get feature importance
    def show_feature_importance(self):
        #fit model to training data
        model_pipeline = self.build_model()
        model_pipeline.fit(self.X_train, self.y_train)
        #1. get preprocessor from pipeline
        preprocessor = model_pipeline.named_steps['preprocessor']
        #2. Get feature names from Column transformer
        feature_names = []
        #3. Extract numerical features
        num_cols = X_train.select_dtypes(include='number').columns.to_list()
        for cols in num_cols:
            feature_names.append(cols)
        #4. Extract categorical features
        cat_cols = X_train.select_dtypes(include= 'category').columns.to_list()
        for cols in cat_cols:
            #Get categories from OneHotEncoder
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            categories = ohe.categories_[0]
            #skip first category due to drop_first
            for cat in categories[1:]:
                feature_names.append(f"{cols}_{cat}")
        #5. Extract model from pipeline
        model = model_pipeline.named_steps['classifier']
        #6. Get Coefficients
        coefficients = model.coef_[0]
        intercept = model.intercept_[0] 
        #create dataframe with feature importance
        feature_importance = pd.DataFrame(
            {'features':feature_names,
            'coefficients': coefficients,
            'absolute_coefficient': np.abs(coefficients)})
        #sort values in ascending order
        feature_importance = feature_importance.sort_values(by='absolute_coefficient', ascending=True)
        plt.barh(feature_importance['features'], feature_importance['absolute_coefficient'])
        plt.title('Feature Importance Using Coefficients')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Features')
        plt.show()

    #save model
    def dump(self):
        #fit model to training data
        model_pipeline = self.build_model()
        model_pipeline.fit(self.X_train, self.y_train)
        #create filepath to save and store model
        filepath = os.path.join(self.model_directory, self.model_subdirectory)
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        #save model
        joblib.dump(model_pipeline, filepath)
        return filepath

    #load model
    def load(self):
        #prepare a pattern for the glob search
        pattern = os.path.join(self.model_directory, self.model_subdirectory)
        try:
            model_path = sorted(glob(pattern))[-1]
        except IndexError:
            raise Exception(f"Oops No model trained for {self.model_subdirectory} chai..") 
        self.model = joblib.load(model_path)
        return self.model