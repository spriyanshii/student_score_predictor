
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


df = pd.read_csv("students_score.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

plt.figure(figsize=(6,4))
sns.scatterplot(x='Hours', y='Scores', data=df)
plt.title("Study Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Score (%)")
plt.show()

X = df[['Hours']]
y = df['Scores']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted Scores:")
print(compare)


joblib.dump(model, 'student_score_model.pkl')
print("\nModel saved as 'student_score_model.pkl'")


plt.figure(figsize=(6,4))
sns.regplot(x='Hours', y='Scores', data=df, line_kws={'color':'red'})
plt.title('Regression Line - Hours vs Scores')
plt.xlabel("Hours Studied")
plt.ylabel("Score (%)")
plt.show()

# Predict for a new value
hours = 9.25
predicted_score = model.predict([[hours]])
print(f"\nIf a student studies {hours} hours, predicted score is {predicted_score[0]:.2f}%")

