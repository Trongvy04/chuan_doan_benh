from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

data = pd.read_csv("./data/dataset1.csv")

X = data.drop("Disease", axis=1)
y = data["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = RandomForestClassifier(n_estimators=50, random_state=1, class_weight='balanced')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

result_df = pd.DataFrame({'y': y_test, 'y_predict': y_pred})
print(result_df.sample(10))


print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.2f}")

joblib.dump(model, "disease_prediction_model.pkl")
print("✅ Mô hình đã được lưu vào 'disease_prediction_model.pkl'")