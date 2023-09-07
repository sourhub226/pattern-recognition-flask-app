from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from scipy.stats import multivariate_normal

# import matplotlib.pyplot as plt
import numpy as np
import pickle
from io import BytesIO


app = Flask(__name__)
app.secret_key = "pr1"


@app.route("/")
def index():
    return render_template("home.html")


def store_in_session(key, value):
    if key in session:
        session.pop(key)
    session[key] = value


@app.route("/stats", methods=["GET", "POST"])
def stats():
    class_stats = {}
    if request.method == "POST":
        file = request.files["dataset"]
        df = pd.read_csv(BytesIO(file.read()))
        store_in_session("dataset", pickle.dumps(df))

    dataset_stats = {
        "no_rows": len(df),
        "no_num_features": len(df.select_dtypes(include=["number"]).columns),
    }
    dataset_stats["no_cat_features"] = (
        len(df.select_dtypes(exclude=["number"]).columns) - 1
    )

    unique_classes = df["Class"].unique()

    for class_label in unique_classes:
        classviz_df = df[df["Class"] == class_label].select_dtypes(include=["number"])
        class_stats[class_label] = {
            "description": classviz_df.describe().to_html(),
            "kurtosis": classviz_df.kurt().to_frame(name="kurtosis").T.to_html(),
            "skewness": classviz_df.skew().to_frame(name="skewness").T.to_html(),
            "covariance": classviz_df.cov(ddof=0).to_html(),
        }

        # display graph using matplotlib
        # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # axs = axs.ravel()
        # x, y = np.meshgrid(
        #     np.linspace(class_df["Height"].min(), class_df["Height"].max(), 100),
        #     np.linspace(class_df["Weight"].min(), class_df["Weight"].max(), 100),
        # )
        # pos = np.dstack((x, y))
        # axs[iteration].contourf(x, y, mvnd.pdf(pos), cmap="viridis")
        # axs[iteration].set_title(f"Class {class_label} Distribution")
        # img_buffer = io.BytesIO()
        # plt.tight_layout()
        # plt.savefig(img_buffer, format="png")
        # img_buffer.seek(0)
        # plot_data = base64.b64encode(img_buffer.read()).decode("utf-8")

    return render_template(
        "stats.html",
        filename=file.filename,
        df_sample=df.head(10).to_html(
            index=False,
            justify="left",
        ),
        dataset_stats=dataset_stats,
        class_stats=class_stats,
        # plot_data=plot_data,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    pdf = {}
    max_probability = -1
    predicted_class = None
    data_point = None
    df = pickle.loads(session.get("dataset"))
    if request.method == "POST":
        data_point = request.form["data_point"]
        data_point = np.array([[float(x) for x in data_point.split(",")]])
        print(data_point)

    unique_classes = df["Class"].unique()
    for class_label in unique_classes:
        classviz_df = df[df["Class"] == class_label].select_dtypes(include=["number"])

        probability = multivariate_normal(
            mean=classviz_df.mean(), cov=classviz_df.cov()
        ).pdf(data_point)
        pdf[class_label] = {"probability": probability}
        print(class_label)
        print(probability)
        if probability > max_probability:
            max_probability = probability
            predicted_class = class_label
        print(f"Predicted class = {predicted_class}")

    return render_template(
        "predict.html",
        df=df,
        predicted_class=predicted_class,
        feature_cols=df.select_dtypes(include=["number"]).columns.tolist(),
        pdf=pdf,
    )


if __name__ == "__main__":
    app.run(debug=True)
