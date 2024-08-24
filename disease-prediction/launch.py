import os
from dotenv import load_dotenv

import pandas as pd
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from disease_prediction.analysis import analyze_the_data
from disease_prediction.model import (normalize_data, train_the_model,
                                      get_accuracy_test_data)
from disease_prediction.visualization import (visualize_ratio, building_pair,
                                              building_correlation_matrix,
                                              visualize_balanced_ratio,
                                              visualize_confusion_matrix)

load_dotenv()

PATH_TO_FILE = os.getenv('PATH_TO_FILE')
df = pd.read_csv(PATH_TO_FILE)


def to_run():
    # data analyze
    analyze_the_data(df)

    # visualization
    print('\nWe visualize the ratio of the state of health in the dataset\n')
    visualize_ratio(df)
    print('\nBuilding paired diagrams for some features\n')
    building_pair(df)
    print('\nBuilding a heat map for the correlation matrix\n')
    building_correlation_matrix(df)

    # data normalization
    x_resampled, y_resampled = normalize_data(df)

    # the ratio of the state of health in the dataset after balancing
    visualize_balanced_ratio(y_resampled)

    # data separation into training and test sets (80% / 20%)
    x_train, x_test, y_train, y_test = train_test_split(x_resampled,
                                                        y_resampled,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y_resampled)

    # train the model
    best_model = train_the_model(x_train, y_train)
    y_pred = best_model.predict(x_test)

    # prediction based on a test sample
    get_accuracy_test_data(y_pred, y_test)

    # visualization of the error matrix (confusion matrix)
    visualize_confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))
