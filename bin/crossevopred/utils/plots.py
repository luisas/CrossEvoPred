import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_loss_vs_epoch(infos, suptitle = "", title = "Loss vs epoch"):
    """
    Plot loss vs epoch
    :param training_infos: dict, training information
    :param suptitle: str,
    :param title: str
    :return: plot
    """
    plot_3 = plt.figure(figsize=(5, 5))  # Adjust width and height as needed
    plot_3 = sns.lineplot(data=infos, x="epoch", y="loss")
    plt.xticks(np.arange(1, len(infos["epoch"])+1, 1.0))
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylim(0, max(infos["loss"]) + 10 * np.std(infos["loss"]))
    plt.title(title)
    sns.despine(offset=10, trim=True)
    plt.suptitle(suptitle)
    return plot_3