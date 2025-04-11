import matplotlib.pyplot as plt

def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", legend=True):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bo", label="Target")
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "rx", markersize=10, label="Prediction")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    if legend and (y or y_pred):
        plt.legend(fontsize=14, loc="upper left")

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
    for col in range(3):
        plt.sca(axes[col])
        plot_series(X_valid[col, :, 0], y_valid[col, 0],
                    y_label=("$x(t)$" if col==0 else None),
                    legend=(col == 0))
    save_fig("time_series_plot")
    plt.show()