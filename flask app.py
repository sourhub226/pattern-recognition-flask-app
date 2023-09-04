from flask import Flask, render_template, request, redirect, url_for
import pandas as pd


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():
    class_stats = dataset_stats = ""
    file = request.files["dataset"]
    df = pd.read_csv(file.filename or "bmi.csv")
    dataset_stats += f"Total no of rows = {len(df)} <br/>"
    dataset_stats += f"Total no of numeric features = {len(df.select_dtypes(include=['number']).columns) } <br/>"
    dataset_stats += f"Total no of categorical features = {len(df.select_dtypes(exclude=['number']).columns)-1}"

    unique_classes = df["Class"].unique()

    for class_label in unique_classes:
        class_df = df[df["Class"] == class_label]
        stats1_df = class_df.describe()
        stats2_df = pd.DataFrame(
            {
                "Height": [class_df["Height"].kurt(), class_df["Height"].skew()],
                "Weight": [class_df["Weight"].kurt(), class_df["Weight"].skew()],
            },
            index=["kurtosis", "skewness"],
        )

        class_stats += f"For <b>{class_label}</b>:"
        class_stats += pd.concat(
            [stats1_df, stats2_df],
        ).to_html()
        class_stats += "<br/>"

    return render_template(
        "stats.html",
        filename=file.filename,
        df=df.head(10).to_html(
            index=False,
            justify="left",
        ),
        dataset_stats=dataset_stats,
        class_stats=class_stats,
    )


if __name__ == "__main__":
    app.run(debug=True)
