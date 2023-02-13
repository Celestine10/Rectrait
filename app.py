#####################################################

import pandas as pd
from flask import Flask, render_template, request
from numpy import *
from sklearn import linear_model

#####################################################

app = Flask(__name__)

#####################################################
# Loading Dataset Globally
data = pd.read_csv("person.csv")
array = data.values

#Convert male and female category into 1 and 0
for i in range(len(array)):
    if array[i][0] == "Male":
        array[i][0] = 1
    else:
        array[i][0] = 0

df = pd.DataFrame(array)

maindf = df[[0, 1, 2, 3, 4, 5, 6]]
mainarray = maindf.values

temp = df[7]
#y_train = temp.values
y_train = temp.values

for i in range(len(y_train)):
    y_train[i] = str(y_train[i])


mul_lr = linear_model.LogisticRegression(
    multi_class="ovr", solver="lbfgs", max_iter=100, penalty="l2")
mul_lr.fit(mainarray, y_train)
#####################################################


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    else:
        age = int(request.form["age"])
        if age < 17:
            age = 17
        elif age > 70:
            age = 70

        inputdata = [
            [
                request.form["gender"],
                age,
                9 - int(request.form["openness"]),
                9 - int(request.form["neuroticism"]),
                9 - int(request.form["conscientiousness"]),
                9 - int(request.form["agreeableness"]),
                9 - int(request.form["extraversion"]),
            ]
        ]

        for i in range(len(inputdata)):
            if inputdata[i][0] == "Male":
                inputdata[i][0] = 1
            else:
                inputdata[i][0] = 0

        df1 = pd.DataFrame(inputdata)
        testdf = df1[[0, 1, 2, 3, 4, 5, 6]]
        maintestarray = testdf.values

        y_pred = mul_lr.predict(maintestarray)
        
        for i in range(len(y_pred)):
            y_pred[i] = str((y_pred[i]))
        DF = pd.DataFrame(y_pred, columns=["Predicted Personality"])
        DF.index = DF.index + 1
        DF.index.names = ["Person No"]

        return render_template(
            "result.html", per=DF["Predicted Personality"].tolist()[0]
        )


@app.route("/learn")
def learn():
    return render_template("learn.html")


@app.route("/working")
def working():
    return render_template("working.html")


# Handling error 404
@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", code=404, text="Page Not Found"), 404


# Handling error 500
@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", code=500, text="Internal Server Error"), 500


if __name__ == "__main__":

    # use 0.0.0.0 for replit hosting
    app.run(host="0.0.0.0", port=8080)

    # for localhost testing
    # app.run()
