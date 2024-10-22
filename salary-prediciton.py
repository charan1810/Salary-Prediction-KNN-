import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st


st.title('Salary Prediction ')

dataset=pd.read_csv(r"emp_sal.csv")




# Bar chart for Position vs Salary
st.write("### Salary vs Position Bar Chart")
fig, ax = plt.subplots()
ax.bar(dataset['Position'], dataset['Salary'], color='skyblue')
ax.set_xlabel('Position')
ax.set_ylabel('Salary')
ax.set_title('Salary vs Position')
plt.xticks(rotation=45)
st.pyplot(fig)  # Display the chart in Streamlit

st.write("Salary v/s Position Bar Chart")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
regressor=KNeighborsRegressor(n_neighbors=5, weights='distance', p=2)
regressor.fit(x,y)

st.write("### Enter a position level to predict the salary from (0-10)")
user_input=st.number_input("")

knn_pred=regressor.predict([[user_input]])

st.write(f"### Predicted salary for position level {user_input}:")
st.write(f"₹{knn_pred[0]:,.2f}")
