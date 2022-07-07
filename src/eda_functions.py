import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# set the plot style
sns.set_theme(context="notebook", style="darkgrid", palette="crest", font="helvetica")
cmap = sns.color_palette("crest")
sns.set(rc = {"figure.dpi":300})
sns.set(rc = {"figure.figsize":(6,3)})
sns.set(font_scale = 0.5)


def get_cloudmask_frame(df:pd.DataFrame, column:str) -> pd.DataFrame:
    """ Create a data frame which is easy to plot with the number of observations 
        for the unclouded or no information state for a given column of the data frame.

    Args:
        df (pd.DataFrame): Data frame that contains the CLM and at least another column to count the entries.
        column (str): The column to be counted.

    Returns:
        pd.DataFrame: Data frame that is ready to plot.
    """
    # create two series with the number of observations that are unclouded or have no CLM information 
    unclouded = df[df["CLM"]==0].groupby(column)[column] \
    .count()
    noinfo = df[df["CLM"]==255].groupby(column)[column] \
    .count()
    # create a data frame of the two series
    cloudmask_df = pd.concat([unclouded, noinfo], axis=1)
    cloudmask_df.columns = ["unclouded", "no information"]
    cloudmask_df = cloudmask_df.reset_index()
    # melts the data frame in order to make it plot-able
    cloudmask_df = cloudmask_df.melt(id_vars=column).rename(columns=str.title)
    cloudmask_df.columns = [column.title(), "CLM", "Count"]

    return cloudmask_df


def get_info_per_field_id(df:pd.DataFrame, column:str) -> pd.DataFrame:
    """ Create a data frame with the amount of observations 
        for the given column per field_id.

    Args:
        df (pd.DataFrame):  Data frame containing the field_id and 
                            at least another column to count the observations.
        column (str):       The column to count the observations for. 

    Returns:
        pd.DataFrame: Data frame with the count per field_id.
    """
    # create a list of series with the occurrence of field_ids per month 
    counts = []
    for entry in sorted(df[column].unique()):
        count_perID = df[df[column]==entry].groupby("field_id")["field_id"].count()
        counts.append(count_perID)
    # create a data frame of list of series
    perID_df = pd.concat(counts, axis=1)
    perID_df.columns = sorted(df[column].unique())
    # melt the data frame
    perID_df = perID_df.reset_index().melt(id_vars="field_id")
    perID_df.columns = ["Field_ID", column.title(), "Count"]
    return perID_df


def plot_confusion_matrix(xgb_cm:confusion_matrix, labels:list):
    """Plots a nicely formatted confusion matrix. 

    Args:
        xgb_cm (sklearn.metrics.confusion_matrix): The confusion matrix to be plotted.
        labels (list): List of the class label names.
    """

    # setup the heatmap of the confusion matrix
    ax = sns.heatmap(xgb_cm, annot=True, cmap=cmap, fmt="g")
    ax.set_title("Confusion Matrix with labels\n");
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values ");

    # set the ticket labels
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90, horizontalalignment="center") 

    # display the confusion matrix
    plt.show()


def get_label_accuracies(xgb_cm, labels):
    # create empty lists to catch the results
    accuracies = []
    often_confused_with = []
    confusion_percent = []

    for idx, label in enumerate(labels):
      # calculate the accuracy for the current label
      highest_value = np.diagonal(xgb_cm)[idx]
      sum_of_labels = np.sum(xgb_cm[idx])
      label_accuracy = round(highest_value/sum_of_labels, 2)
      accuracies.append(label_accuracy)

      # identify the label that the current one is most often confused with
      second_highest_value = np.partition(xgb_cm[idx].flatten(), -2)[-2]
      second_highest_value_idx = (np.where(xgb_cm[idx] == second_highest_value))
      often_confused_with.append(labels[int(second_highest_value_idx[0])])

      # get the confusion in percentage
      confusion = round(second_highest_value/highest_value*100, 2)
      confusion_percent.append(confusion)

    # create a data frame using a dictionary
    data = {
        "crop_type":labels, 
        "accuracy":accuracies,
        "often_confused_with":often_confused_with,
        "confusion_percent":confusion_percent
        }
    acc_df = pd.DataFrame(data)

    # sort the data frame in order decreasing feature importance
    acc_df.sort_values(by=["accuracy"], ascending=False, inplace=True)

    return acc_df


def plot_label_accuracy(df):
    # plot Searborn bar chart
    ax = sns.barplot(data=df, x="crop_type", y="accuracy", color=cmap[4])

    # do the annotation on each bar
    for bar in ax.patches:
        ax.annotate(format(bar.get_height(), ".2f"), 
                      (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                      ha = "center", va = "center", 
                      xytext = (0, 9), 
                      textcoords = "offset points")

    # set chart labels
    ax.set_title("Accuracy of the different crop types\n")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")

    # set the ticket labels
    plt.yticks(rotation=0) 
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90, horizontalalignment="center")

    # display the bar chart of the label accuracies
    plt.show()


def plot_feature_importance(importance, feature_names, model_name, num_features=None):
    # create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(feature_names)

    # create a data frame using a dictionary
    data={"feature_names":feature_names, "feature_importance":feature_importance}
    fi_df = pd.DataFrame(data)

    # sort the data frame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    
    # set the number of features to display
    fi_df = fi_df.head(num_features)
    
    # define size of bar plot
    plt.figure(figsize=(10,8))

    # plot Searborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"], color=cmap[4])

    # set chart labels
    plt.title(f"{model_name} Feature Importance\n")
    plt.xlabel("\nFeature importance")
    plt.ylabel("Feature names")
