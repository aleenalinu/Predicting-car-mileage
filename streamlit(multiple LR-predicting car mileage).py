import streamlit as st
st.set_page_config(layout="wide")
st.header("Multiple Linear Regression")
st.title("Problem Statement: Predicting Car Mileage (Fuel Efficiency)")
st.write("Fuel efficiency (measured as mileage or miles per gallon) is one of the most important performance indicators for vehicles. It affects running costs, environmental impact, and vehicle design choices. Manufacturers and consumers alike are interested in understanding which factors influence a car’s mileage and by how much. To explore these relationships, a dataset has been created that includes various numerical attributes of cars — such as engine size, weight, horsepower, age, and tire pressure — and their corresponding mileage")
st.write("The goal of this project is to develop a multiple linear regression model that predicts a car’s mileage based on its physical and mechanical characteristics")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("car_mileage_dataset.csv")
show_data = st.checkbox("Show Dataset")

if show_data:
    st.dataframe(data)
# ---------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------  
data=data.drop_duplicates()
data.duplicated().sum()
data=data.dropna()
data.isna().sum()

st.subheader("Outlier Analysis")
fig=plt.figure(figsize=(10,10))
plt.suptitle("Outlier Analysis")
# Fig 1
plt.subplot(6,1,1)
plt.title("Engine_Size")
sns.boxplot(x=data['Engine_Size'],color='lightblue',linecolor='black')
# Fif 2
plt.subplot(6,1,2)
plt.title("Weight")
sns.boxplot(x=data['Weight'],color='lightblue',linecolor='black')
# Fif 2
plt.subplot(6,1,3)
plt.title("Horsepower")
sns.boxplot(x=data['Horsepower'],color='lightblue',linecolor='black')
# Fif 2
plt.subplot(6,1,4)
plt.title("Age")
sns.boxplot(x=data['Age'],color='lightblue',linecolor='black')
# Fif 2
plt.subplot(6,1,5)
plt.title("Tire_Pressure")
sns.boxplot(x=data['Tire_Pressure'],color='lightblue',linecolor='black')
# Fif 2
plt.subplot(6,1,6)
plt.title("Mileage")
sns.boxplot(x=data['Mileage'],color='lightblue',linecolor='black')

plt.tight_layout()
st.pyplot(fig)

# ---------------------------------------------------------
# Outlier Removal
# ---------------------------------------------------------
data=data[(data['Weight']>=2000) & (data['Weight']<=4000)]
data=data[data["Mileage"]<50]
ata=data.reset_index(drop=True)

st.subheader("EDA")
# Set figure size for the grid
fig=plt.figure(figsize=(18, 12))

# 1. Heatmap (Correlation analysis)
plt.subplot(2, 3, 1)
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")

# 2. Histogram (Distribution of Mileage)
plt.subplot(2, 3, 2)
sns.histplot(data['Mileage'], kde=True, color='blue')
plt.title("Distribution of Mileage")

# 3. Scatter Plot (Engine Size vs Mileage)
plt.subplot(2, 3, 3)
sns.scatterplot(x='Engine_Size', y='Mileage', data=data, color='green')
plt.title("Engine Size vs Mileage")

# 4. Line Plot (Age vs Mileage trend)
plt.subplot(2, 3, 4)
sns.lineplot(x='Age', y='Mileage', data=data, color='red')
plt.title("Age vs Mileage Trend")

# 5. Bar Plot (Average Horsepower by Engine Size)
plt.subplot(2, 3, 5)
sns.barplot(x='Engine_Size', y='Horsepower', data=data)
plt.title("Avg Horsepower by Engine Size")

# 6. Regplot (Linear relationship: Weight vs Mileage)
plt.subplot(2, 3, 6)
sns.regplot(x='Weight', y='Mileage', data=data, color='purple')
plt.title("Weight vs Mileage Linear Regression")

plt.tight_layout()
plt.show()
st.pyplot(fig)

# ---------------------------------------------------------
# Model Building
# ---------------------------------------------------------
st.subheader("Model Building")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
# Independent variables
X = data[['Engine_Size', 'Weight', 'Horsepower', 'Age', 'Tire_Pressure']]

# Dependent variable
y = data['Mileage']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------
st.subheader("Feature Scaling")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Linear Regression Model
# ---------------------------------------------------------
model = LinearRegression()

model.fit(X_train_scaled, y_train)

st.write("Model Intercept:", model.intercept_)

st.write("Model Coefficients:")

coefficient = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(coefficient, use_container_width=True)

# ---------------------------------------------------------
# Prediction on Test Data
# ---------------------------------------------------------
st.subheader("Model Prediction")

y_pred = model.predict(X_test_scaled)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.dataframe(comparison, use_container_width=True)

# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------
st.subheader("Model Evaluation")

mape = mean_absolute_percentage_error(y_test, y_pred) * 100

st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")

# ---------------------------------------------------------
# Actual vs Predicted Plot
# ---------------------------------------------------------
st.subheader("Actual vs Predicted")

fig = plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Mileage")
plt.ylabel("Predicted Mileage")
plt.title("Actual vs Predicted Mileage")

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color='red'
)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# Predict Mileage for New Car
# ---------------------------------------------------------
st.subheader("Predict Mileage for New Car")

st.write("Enter the car details:")

col1, col2 = st.columns(2)

with col1:
    engine_size = st.number_input(
        "Engine Size",
        value=1.5
    )

    weight = st.number_input(
        "Weight",
        value=3000.0
    )

    horsepower = st.number_input(
        "Horsepower",
        value=130.0
    )

with col2:
    age = st.number_input(
        "Age",
        value=10.0
    )

    tire_pressure = st.number_input(
        "Tire Pressure",
        value=28.0
    )

new = pd.DataFrame({
    'Engine_Size': [engine_size],
    'Weight': [weight],
    'Horsepower': [horsepower],
    'Age': [age],
    'Tire_Pressure': [tire_pressure]
})

if st.button("Predict Mileage"):

    new_scaled = scaler.transform(new)

    prediction = model.predict(new_scaled)

    st.success(
        f"Predicted Mileage: {prediction[0]:.2f} MPG"
    )
