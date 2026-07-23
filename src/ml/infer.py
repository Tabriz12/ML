import pandas as pd

from ml.error import MSE
from ml.models import LinearRegression

data = pd.read_csv(
    "data/Student_Marks.csv",
)

X: pd.DataFrame = data[["number_courses", "time_study"]]

Y: pd.Series = data["Marks"]

x_train, y_train = X.loc[:90], Y.loc[:90]
x_test, y_test = X.loc[90:], Y.loc[90:]

linear = LinearRegression()

linear.fit(x_train, y_train, 5000, lr=1e-3)

print(linear.B, linear.W)

predictions = linear.predict(x_test)

print(MSE(y_test, predictions))
