import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import numpy as np


def create_pairwise_scatter_plot(df, features):
    sns.pairplot(
        df[features],
        diag_kind="kde",
        plot_kws={"alpha": 0.25, "s": 5, "edgecolor": "k"},
    )
    plt.tight_layout()
    plt.show()


def create_correlation_matrix(df, features):
    corr_matrix = df[features].corr()
    plt.figure(figsize=(15, 13))
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="RdBu")
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.show()


def plot_backtest_results(results, predictions):
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.5))
    sns.histplot(results[f"corr_test"], ax=axs[0], kde=True, bins=15)
    sns.histplot(results[f"r2_test"], ax=axs[1], kde=True, color="red", bins=15)
    axs[0].set_ylabel("")
    axs[1].set_ylabel("")

    corr_test_mean, corr_test_std = (
        round(results[f"corr_test"].mean(), 2),
        round(results[f"corr_test"].std(), 2),
    )
    axs[0].set_title(f"corr (test)")
    axs[0].set_xlabel(f"Mean: {corr_test_mean} ± {corr_test_std}")
    r2_test_mean, r2_test_std = round(results[f"r2_test"].mean(), 2), round(
        results[f"r2_test"].std(), 2
    )
    axs[1].set_title(f"r2 (Test)")
    axs[1].set_xlabel(f"Mean: {r2_test_mean} ± {r2_test_std}")

    sns.lineplot(
        x=results.index,
        y=results[f"corr_test"],
        ax=axs[2],
        color="blue",
        alpha=0.3,
        label="corr",
    )
    sns.lineplot(
        x=results.index,
        y=results[f"r2_test"],
        ax=axs[2],
        color="red",
        alpha=0.3,
        label="r2",
    )
    sns.scatterplot(
        x=results.index,
        y=results[f"corr_test"],
        ax=axs[2],
        color="blue",
        alpha=1,
        s=20,
    )
    sns.scatterplot(
        x=results.index,
        y=results[f"r2_test"],
        ax=axs[2],
        color="red",
        alpha=1,
        s=20,
    )
    axs[2].set_xlabel("Days")
    axs[2].set_ylabel(f"")
    axs[2].legend()
    axs[2].set_title(f"Corr / R2 vs Days")
    sns.scatterplot(
        x=predictions["y_pred"],
        y=predictions["y_real"],
        ax=axs[3],
        alpha=0.25,
        s=4,
        c="red",
    )
    axs[3].set_title("Predicted vs Actual")
    axs[3].set_ylabel("")
    axs[3].set_xlabel(
        f"R2: {round(r2_score(predictions["y_real"], predictions["y_pred"]), 3)} | Corr: {round(np.corrcoef(predictions["y_real"], predictions["y_pred"])[0, 1], 3)}"
    )
    plt.show()


def plot_feature_importances(df, train_features, output_feature, tscv, top_n=10):
    dates = df['time'].dt.date
    linear_importances, rf_importances = (
        np.zeros(len(train_features)),
        np.zeros(len(train_features)),
    )
    for train_index, test_index in tscv.split(dates):
        train_dates, test_dates = (
            dates[train_index], 
            dates[test_index]
        )
        train_df, test_df = (
            df[df['time'].dt.date.isin(train_dates)], 
            df[df['time'].dt.date.isin(test_dates)]
        )
        ridge = Ridge(alpha=0.1)
        ridge.fit(train_df[train_features], train_df[output_feature])
        linear_importances += ridge.coef_

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_df[train_features], train_df[output_feature])
        rf_importances += rf.feature_importances_

    