from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


def normalize_data(df):
    # We specify the labels
    x = df.drop(['name', 'status'], axis=1)
    y = df['status']

    # normalize the signs
    scaler = MinMaxScaler((-1, 1))
    x_normalized = scaler.fit_transform(x)

    # we balance the number of examples of both classes
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x_normalized, y)
    return x_resampled, y_resampled


def train_the_model(x_train, y_train):
    # we define ranges of values for various hyperparameters
    param_grid = {'n_estimators': [100, 200, 300],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'max_depth': [3, 5, 7], 'subsample': [0.8, 0.9, 1.0],
                  'colsample_bytree': [0.8, 0.9, 1.0], 'gamma': [0, 0.1, 0.2],
                  'reg_alpha': [0, 0.01, 0.1], 'reg_lambda': [0.5, 1.0, 1.5]}

    # creating a basic model
    model = XGBClassifier(random_state=42)

    # configure GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', n_jobs=-1, cv=5, verbose=1)

    # training the model
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def get_accuracy_test_data(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy based on test data: {accuracy * 100:.2f}%")
