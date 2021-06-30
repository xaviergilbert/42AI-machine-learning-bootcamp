import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# def confusion_matrix(y, y_hat):


def accuracy_score_(y, y_hat):
    """
        Compute the accuracy score.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
        Returns: 
            The accuracy score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    res = (tp + tn) / (tn + fp + fn + tp)
    return res

def precision_score_(y, y_hat, pos_label=1):
    """
        Compute the precision score.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report the precision_score (default=1)
        Returns: 
            The precision score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """
    # y_hat = np.append(np.array([pos_label]), y_hat)
    # y = np.append(np.array([pos_label]), y)
    # print(y_hat)
    y = [1 if yi == pos_label else 0 for yi in y]
    y_hat = [1 if y_hati == pos_label else 0 for y_hati in y_hat]
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    # tp -= 1
    res = tp / (tp + fp)
    return res


def recall_score_(y, y_hat, pos_label=1):
    """
        Compute the recall score.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report the precision_score (default=1)
        Returns: 
            The recall score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """
    y = [1 if yi == pos_label else 0 for yi in y]
    y_hat = [1 if y_hati == pos_label else 0 for y_hati in y_hat]
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    res = tp / (tp + fn)
    return res


def f1_score_(y, y_hat, pos_label=1):
    """
        Compute the f1 score.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report the precision_score (default=1)
        Returns: 
            The f1 score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """
    y = [1 if yi == pos_label else 0 for yi in y]
    y_hat = [1 if y_hati == pos_label else 0 for y_hati in y_hat]
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    precision = precision_score_(y, y_hat)
    recall = recall_score_(y, y_hat)
    res = (2 * precision * recall) / (precision + recall)
    return res

if __name__ == "__main__":
    # Example 1:
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])

    # Accuracy
    ## your implementation
    print("Accuracy : ", accuracy_score_(y, y_hat))
    ## sklearn implementation
    print("Accuracy sklearn: ", accuracy_score(y, y_hat))
    print()

    # Precision
    ## your implementation
    print("Precision : ", precision_score_(y, y_hat))
    ## sklearn implementation
    print("Precision sklearn : ", precision_score(y, y_hat))
    print()

    # Recall
    ## your implementation
    print("Recall score : ", recall_score_(y, y_hat))
    ## sklearn implementation
    print("Recall score sklearn : ", recall_score(y, y_hat))
    print()

    # F1-score
    ## your implementation
    print("f1 score : ", f1_score_(y, y_hat))
    ## sklearn implementation
    print("f1 score sklearn : ", f1_score(y, y_hat))
    print()

    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    # Accuracy
    ## your implementation
    print("Accuracy : ", accuracy_score_(y, y_hat))
    ## sklearn implementation
    print("Accuracy sklearn: ", accuracy_score(y, y_hat))
    print()

    # Precision
    ## your implementation
    print("Precision : ", precision_score_(y, y_hat, pos_label='dog'))
    ## sklearn implementation
    print("Precision sklearn : ", precision_score(y, y_hat, pos_label='dog'))
    print()

    # Recall
    ## your implementation
    print("Recall : ", recall_score_(y, y_hat, pos_label='dog'))
    ## sklearn implementation
    print("Recall sklearn : ", recall_score(y, y_hat, pos_label='dog'))
    print()

    # F1-score
    ## your implementation
    print("f1 score : ", f1_score_(y, y_hat, pos_label='dog'))
    ## sklearn implementation
    print("f1 score : ", f1_score(y, y_hat, pos_label='dog'))
    print()

    # Example 3:
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    # Accuracy
    ## your implementation
    print("Accuracy : ", accuracy_score_(y, y_hat))
    ## sklearn implementation
    print("Accuracy sklearn : ", accuracy_score(y, y_hat))
    print()

    # Precision
    ## your implementation
    print("Precision : ", precision_score_(y, y_hat, pos_label='norminet'))
    ## sklearn implementation
    print("Precision sklearn : ", precision_score(y, y_hat, pos_label='norminet'))
    print()

    # Recall
    ## your implementation
    print("Recall score : ", recall_score_(y, y_hat, pos_label='norminet'))
    ## sklearn implementation
    print("Recall score sklearn : ", recall_score(y, y_hat, pos_label='norminet'))
    print()

    # F1-score
    ## your implementation
    print("f1 score : ", f1_score_(y, y_hat, pos_label='norminet'))
    ## sklearn implementation
    print("f1 score sklearn : ", f1_score(y, y_hat, pos_label='norminet'))
    exit()