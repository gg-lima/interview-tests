# Fuse Data Scientist Work Sample

Interview test: for more details about the test and analysis results, see DS_Worksample.ipynb.

# Directory Structure

```
├── README.md
├── data/ : The directory with data.
│   ├── example-grades-to-predict.csv : csv file with example data to predict.
│   └── grades.csv : csv file with data to train model.
│
├── models/ : The directory with trained models.
│   └── baseline_regression_linear.pickle : trained model.
│
├── DS_Worksample.ipynb: Jupyter notebook with analysis.
├── docker-compose.yml : Docker container config.
├── Dockerfile : Docker file.
├── evaluate.py : script to evaluate the model.
├── predict.py : script to predict final grades.
└── requirements.txt : The requirements file for reproducing the analysis environment
```

# Environment

Python version 3.8.5

## Docker

Using docker to reproduce the analysis environment:

```
docker-compose up -d
```

After create the container, to access the Jupyter Notebook with analysis.

```
http://0.0.0.0:8888
or
http://localhost:8888
```

# Model Usage:

If you are using Docker to reproduce the analysis environment, before to run the python scripts create a Bash session in the container:

```
docker exec -it cardinal bash
```

## Prediction

```
python3 -m predict --data '/path/to/data.csv' --model '/path/to/model.pickle' --save '/path/to/save.csv'
```

## Evaluation

```
python3 -m evaluate --data '/path/to/data.csv' --model '/path/to/model.pickle'
```

## Command Line Args Reference
```
predict.py:
    --data: Directory of Pandas DataFrame
      (default: './data/grades.csv')
    --model: Directory of Model
      (default: './models/baseline_regression_linear.pickle')
    --save: Directory to save the predictions
      (default: './data/grades_predicted.csv')

evaluate.py
    --data: Directory of Pandas DataFrame
      (default: './data/grades.csv')
    --model: Directory of Model
      (default: './models/baseline_regression_linear.pickle')
```