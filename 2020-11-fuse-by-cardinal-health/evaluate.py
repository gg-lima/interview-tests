''' Script to evaluate model given a dataset.
'''
import pandas as pd
import numpy as np
from statsmodels.iolib.smpickle import load_pickle
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)
import argparse
FLAGS = None

def preprocess_data(df):
    ''' Preprocessing data for model usage.

        Expected dataframe format:

        Student_ID  : <np.int64>    The ID of the student
        Test_1_Grade: <np.float64>  The grade the student earned on Test 1.
        Test_2_Grade: <np.float64>  The grade the student earned on Test 2.
        Test_3_Grade: <np.float64>  The grade the student earned on Test 3.
        Test_4_Grade: <np.float64>  The grade the student earned on Test 4.
        Test_5_Grade: <np.float64>  The grade the student earned on Test 5.
        Test_6_Grade: <np.float64>  The grade the student earned on Test 6.
        Final_Grade : <np.float64>  The average of the 6 test grades
        Tutoring    : <np.object>   Tells when the student enrolled in tutoring.

        Tutoring categories avaible: ['Never', 'Test_2', 'Test_3', 'Test_4', 'Test_5', 'Test_6']

        # Arguments:
            df: Pandas DataFrame

        # Returns:
            A Pandas DataFrame with preprocessed data.
    '''
    # Encoder to create variables for model.
    encoder = {
        'Test_2': {'scores': ['Test_1_Grade'],
                   'n_tutoring': 5},
        'Test_3': {'scores': ['Test_1_Grade', 'Test_2_Grade'],
                   'n_tutoring': 4},
        'Test_4': {'scores': ['Test_1_Grade', 'Test_2_Grade', 'Test_3_Grade',],
                   'n_tutoring': 3},
        'Test_5': {'scores': ['Test_1_Grade', 'Test_2_Grade', 'Test_3_Grade', 'Test_4_Grade'],
                   'n_tutoring': 4},
        'Test_6': {'scores': ['Test_1_Grade', 'Test_2_Grade', 'Test_3_Grade', 'Test_4_Grade', 'Test_5_Grade'],
                   'n_tutoring': 1},
        'Never': {'scores': ['Test_1_Grade', 'Test_2_Grade', 'Test_3_Grade', 'Test_4_Grade', 'Test_5_Grade', 'Test_6_Grade'],
                  'n_tutoring': 0}
    }

    def get_n_tutoring_from_pd_dataframe(df):
        return df.apply(lambda row: encoder[row['Tutoring']]['n_tutoring'], axis=1)

    def get_partial_score_from_pd_dataframe(df):
        return df.apply(lambda row: np.mean(row[encoder[row['Tutoring']]['scores']]), axis=1)

    dataset = df.copy()
    dataset['n_tutoring'] = get_n_tutoring_from_pd_dataframe(dataset)
    dataset['partial_score'] = get_partial_score_from_pd_dataframe(dataset)

    # Filtering and renaming
    dataset = (dataset
               .filter(items=['Student_ID', 'partial_score', 'n_tutoring', 'Final_Grade'])
               .rename(columns={'Final_Grade': 'final_grade'}))

    return dataset

if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/grades.csv', help='Directory of Pandas DataFrame')
    parser.add_argument('--model', type=str, default='./models/baseline_regression_linear.pickle', help='Directory of Model')

    FLAGS, unparsed = parser.parse_known_args()

    # Loading the data
    data = pd.read_csv(FLAGS.data)

    # Preprocess the data
    data = preprocess_data(data)

    # Load model
    model = load_pickle(FLAGS.model)

    # Predictions
    predictions = model.get_prediction(exog=data)
    data['final_grade_predicted'] = predictions.predicted_mean
    print('\nData:\n')
    print(data)
    print()
    print(f"Average increase: "
          f"{100*np.mean((data['final_grade_predicted']/data['final_grade'])-1):.3f}%")
    print()

    # Metrics
    print('\nEvaluation Metrics:')
    rmse_stat = np.sqrt(mean_squared_error(data['final_grade'], predictions.predicted_mean))
    mae_stat = mean_absolute_error(data['final_grade'], predictions.predicted_mean)
    print(f'RMSE: Test={rmse_stat:,.4f}')
    print(f' MAE: Test={mae_stat:,.4f}')
    print()
