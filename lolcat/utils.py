from sklearn.metrics import confusion_matrix
import numpy as np


def cm2str(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, format='.2f'):
    """Pretty print for confusion matrixes. Taken from https://gist.github.com/zachguo/10296432."""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    out = "    " + empty_cell + " "
    for label in labels:
        out += "%{0}s".format(columnwidth) % label
    out += "\n"
    for i, label1 in enumerate(labels):
        out += "    %{0}s ".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}{1}".format(columnwidth, format) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            out += cell + " "
        out += "\n"
    return out
