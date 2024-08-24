import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def visualize_ratio(df):
    plt.figure(figsize=(8, 6))
    plt.pie(df['status'].value_counts(),
            labels=df['status'].value_counts().index,
            autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('The ratio of health status')
    plt.show()


def visualize_balanced_ratio(y_resampled):
    plt.figure(figsize=(8, 6))
    plt.pie(y_resampled.value_counts(), autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('"Distribution of the target variable')
    plt.show()


def building_pair(df):
    sns.pairplot(df, vars=['MDVP:Fo(Hz)', 'MDVP:Jitter(%)',
                           'MDVP:Shimmer', 'NHR'],
                 hue='status')
    plt.suptitle('Pairplot of Selected Features', size=12)
    plt.show()


def building_correlation_matrix(df):
    correlation_df = df.drop('name', axis=1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation matrix")
    plt.show()


def visualize_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    axis_labels = ['healthy', 'ill']
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=axis_labels, yticklabels=axis_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')
    plt.show()
