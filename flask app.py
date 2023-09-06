from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():
    class_stats = dataset_stats = ""
    max_probab = -1
    predicted_class = None
    file = request.files["dataset"]
    df = pd.read_csv(file.filename)
    dataset_stats += f"Total no of rows = {len(df)} <br/>"
    dataset_stats += f"Total no of numeric features = {len(df.select_dtypes(include=['number']).columns) } <br/>"
    dataset_stats += f"Total no of categorical features = {len(df.select_dtypes(exclude=['number']).columns)-1}"

    unique_classes = df["Class"].unique()

    for iteration, class_label in enumerate(unique_classes):
        class_df = df[df["Class"] == class_label]
        class_df = class_df[["Height", "Weight"]]
        stats1_df = class_df.describe()
        stats2_df = pd.DataFrame(
            {
                "Height": [class_df["Height"].kurt(), class_df["Height"].skew()],
                "Weight": [class_df["Weight"].kurt(), class_df["Weight"].skew()],
            },
            index=["kurtosis", "skewness"],
        )
        class_mean = class_df.mean()
        class_cov = class_df.cov()

        mvnd = multivariate_normal(mean=class_mean, cov=class_cov)
        # print(mvnd)
        probab = mvnd.pdf([194, 108])
        print(class_label)
        print(probab)
        if probab > max_probab:
            max_probab = probab
            predicted_class = class_label
        print(f"Predicted class = {predicted_class}")

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

        class_stats += f"For <b>{class_label}</b>:"
        class_stats += pd.concat(
            [stats1_df, stats2_df],
        ).to_html(justify="left")
        # class_stats += pd.DataFrame(class_mean, columns=["Mean"]).to_html()
        class_stats += class_cov.to_html()
        class_stats += str(probab)
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
        predicted_class=predicted_class,
        # plot_data=plot_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
