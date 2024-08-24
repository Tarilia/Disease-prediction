### Description:
Disease-Prediction - a project created for the early prediction of Parkinson's disease. Parkinson's disease is a progressive disorder of the central nervous system that affects movement, causing tremors and stiffness. It has 5 stages, and annually more than 1 million people suffer from it in India alone. It is a chronic disease and currently has no cure. It is a neurodegenerative disorder that affects the dopamine-producing neurons in the brain. The project uses the XGBoost classifier.

### Installation:
- download the project
- make install
- Register it in the file .env the PATH_TO_FILE variable, which specifies the path to the dataset

### Run:
- make run

### Visualization:
#### Let's analyze the dataset and predict early-stage Parkinson's disease using XGBoost machine learning algorithm and sklearn library for feature normalization. Visualize the results and generate a report.
#### Downloading and analyzing the dataset
- looking through the first ten entries

[![image.png](https://i.postimg.cc/HLjcXcZs/image.png)](https://postimg.cc/Mc2p8TJh)

- studying the data structure

[![image.png](https://i.postimg.cc/cJjKmx27/image.png)](https://postimg.cc/hJLP4BSh)

- check for any null values

[![image.png](https://i.postimg.cc/Y0NhM198/image.png)](https://postimg.cc/vxBQvx56)

#### Visualization:
- we visualize the ratio of the state of health in the dataset

[![1.png](https://i.postimg.cc/T3t5Kqc4/1.png)](https://postimg.cc/68GpSZgr)

- building paired diagrams for some features

[![2.png](https://i.postimg.cc/C5pz3vFz/2.png)](https://postimg.cc/9R1mTYwh)

- building a heat map for the correlation matrix

[![3.png](https://i.postimg.cc/3r2yXHBp/3.png)](https://postimg.cc/jCddTmBd)

#### Data preparation, creation and training of the XGBoost model
The graph of the ratio of health indicators shows that the data are unevenly distributed. Using SMOTE, we will create additional synthetic examples for a smaller class to balance the number of examples for both classes.

[![4.png](https://i.postimg.cc/282SGjBq/4.png)](https://postimg.cc/0bKv2snx)

To increase the accuracy of the model on the test dataset, you can use hyperparameter tuning with GridSearchCV. Once the best parameters are found, we use them to evaluate the model on the test data.
```
Fitting 5 folds for each of 6561 candidates, totalling 32805 fits
Best parameters: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1.0, 'subsample': 1.0}
Best accuracy: 0.9659574468085106
```

#### The result of the model's work on test data:
```
Accuracy based on test data: 96.61%
```

#### Building an сonfusion matrix

[![u5.png](https://i.postimg.cc/2SkBjXnW/u5.png)](https://postimg.cc/hfYGpsy4)

#### Сlassification report

[![image.png](https://i.postimg.cc/j57DkVYp/image.png)](https://postimg.cc/PpdfPRxy)
