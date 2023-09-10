import base64
from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os

# Set the backend for matplotlib to generate static images of plots without a GUI
matplotlib.use("Agg")


app = Flask(__name__)
app.secret_key = "pr1"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/stats", methods=["GET", "POST"])
def stats():
    class_stats = []
    class_stats_other = {}
    if request.method == "POST":
        file = request.files["dataset"]
        filename = file.filename
        df = pd.read_csv(BytesIO(file.read()))
        session["filepath"] = filename
        session["dataset"] = df
    elif request.method == "GET":
        df = session["dataset"]
        filename = session["filepath"]

    class_column = df.columns[0]
    numeric_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(exclude=["number"]).columns

    dataset_stats = {
        "no_rows": len(df),
        "no_num_features": len(numeric_columns),
        "no_cat_features": len(categorical_columns) - 1,
    }

    unique_classes = df[df.columns[0]].unique()

    for class_label in unique_classes:
        classviz_df = df[df[class_column] == class_label].select_dtypes(
            include=["number"]
        )

        # Calculate mean, std, min, and max for each feature within this class
        class_stats.extend(
            [
                class_label,
                feature,
                classviz_df[feature].mean(),
                classviz_df[feature].std(),
                classviz_df[feature].min(),
                classviz_df[feature].max(),
                classviz_df[feature].quantile(0.25),
                classviz_df[feature].quantile(0.75),
                classviz_df[feature].kurt(),
                classviz_df[feature].skew(),
            ]
            for feature in numeric_columns
        )
        class_stats_other[class_label] = {
            "covariance": classviz_df.cov(ddof=0).to_html(),
        }
        # print(class_stats)

    # Create the statistics DataFrame
    class_stats_df = pd.DataFrame(
        class_stats,
        columns=[
            "Class",
            "Feature",
            "Mean",
            "Std",
            "Min",
            "Max",
            "25%",
            "75%",
            "Kurtosis",
            "Skewness",
        ],
    )

    # Pivot the DataFrame to show the statistics for each feature together
    pivot_df = class_stats_df.pivot(index="Feature", columns="Class").T

    # Display the statistics table
    # print(pivot_df.T)

    return render_template(
        "stats.html",
        filename=filename,
        df_sample=df.head(10).to_html(
            index=False,
            justify="left",
        ),
        dataset_stats=dataset_stats,
        pivot_df=pivot_df.to_html(),
        unique_classes=unique_classes,
        class_stats_other=class_stats_other
        # class_stats=class_stats,
        # plot_data=plot_data,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    results = {}
    max_probability = -1
    predicted_class = None
    df = session["dataset"]
    unique_classes = df[df.columns[0]].unique()
    if request.method == "POST":
        data_point = request.form["data_point"]
        data_point = np.array([[float(x) for x in data_point.split(",")]])
        print(data_point)

        for class_label in unique_classes:
            classviz_df = df[df[df.columns[0]] == class_label].select_dtypes(
                include=["number"]
            )
            mvnd = multivariate_normal(mean=classviz_df.mean(), cov=classviz_df.cov())
            probability = mvnd.pdf(x=data_point)
            cumul_probab = multivariate_normal(
                mean=classviz_df.mean(), cov=classviz_df.cov()
            ).cdf(x=data_point)
            results[class_label] = {
                "probability": probability,
                "cprobability": cumul_probab,
            }
            print(class_label)
            print(probability)
            if probability > max_probability:
                max_probability = probability
                predicted_class = class_label
            print(f"Predicted class = {predicted_class}")

    return render_template(
        "predict.html",
        predicted_class=predicted_class,
        feature_cols=df.select_dtypes(include=["number"]).columns.tolist(),
        results=results,
        unique_classes=unique_classes,
    )


@app.route("/box_plot")
def box_plot():
    df = session["dataset"]
    numeric_columns = df.select_dtypes(include=["number"]).columns
    fig, axes = plt.subplots(
        len(numeric_columns), 1, figsize=(10, 5 * len(numeric_columns))
    )
    for i, feature in enumerate(numeric_columns):
        df.boxplot(column=feature, by=df.columns[0], grid=False, ax=axes[i])
        axes[i].set_title(f"{feature}")

    fig.suptitle("")
    plt.tight_layout()
    # Save the box plots to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return render_template("box_plot.html", plot_data_uri=plot_data_uri)


if __name__ == "__main__":
    app.run(debug=True)
