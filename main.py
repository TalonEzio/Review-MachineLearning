import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from enum import Enum

class ModelType(Enum):
    NAIVE_BAYES = 'Naive Bayes'
    KNN = 'K-Nearest Neighbors'
    DECISION_TREE = 'Decision Tree'

def train_model(data_file, model_type):
    df = pd.read_csv(data_file)

    label_encoders = {}
    for column in ['Đội hình', 'Sân bãi', 'Phong độ', 'Loại cúp', 'Kết quả']:
        le = LabelEncoder()
        le.fit(df[column])
        label_encoders[column] = le
        df[column] = le.transform(df[column])

    X = df[['Đội hình', 'Sân bãi', 'Phong độ', 'Loại cúp']]
    y = df['Kết quả']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model_map = {
        ModelType.NAIVE_BAYES: GaussianNB(),
        ModelType.KNN: KNeighborsClassifier(),
        ModelType.DECISION_TREE: DecisionTreeClassifier()
    }

    if model_type not in model_map:
        print(f'Mô hình {model_type} không hợp lệ.')
        return None, label_encoders

    model = model_map[model_type]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Độ chính xác của mô hình {model_type.value}: {accuracy:.2f}')

    return model, label_encoders


def predict_new_data(model, label_encoders, input_file):
    input_df = pd.read_csv(input_file)

    for column in ['Đội hình', 'Sân bãi', 'Phong độ', 'Loại cúp']:
        input_df[column] = label_encoders[column].transform(input_df[column])

    X_new = input_df[['Đội hình', 'Sân bãi', 'Phong độ', 'Loại cúp']]
    y_new_pred = model.predict(X_new)
    y_new_pred_labels = label_encoders['Kết quả'].inverse_transform(y_new_pred)

    for i, result in enumerate(y_new_pred_labels):
        print(f'Kết quả dự đoán cho dữ liệu {i + 1}: {result}')


if __name__ == '__main__':
    model_type = ModelType.NAIVE_BAYES
    model, label_encoders = train_model('data.csv', model_type)
    if model:
        predict_new_data(model, label_encoders, 'input.csv')

    model_type = ModelType.DECISION_TREE
    model, label_encoders = train_model('data.csv', model_type)
    if model:
        predict_new_data(model, label_encoders, 'input.csv')

    model_type = ModelType.KNN
    model, label_encoders = train_model('data.csv', model_type)
    if model:
        predict_new_data(model, label_encoders, 'input.csv')
