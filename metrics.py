# METRICS PART
def evaluate_metrics(true_labels, predicted_labels):
    """
    Evaluate and print various metrics for the classification.

    Parameters:
    - true_labels: list or array of true class labels.
    - predicted_labels: list or array of predicted class labels.
    """
    tp = fp = tn = fn = 0

    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

def confusion_matrix(true_labels, predicted_labels):
    """
    Compute the confusion matrix.

    Parameters:
    - true_labels: list or array of true class labels.
    - predicted_labels: list or array of predicted class labels.

    Returns:
    - Confusion matrix as a dictionary.
    """
    tp = fp = tn = fn = 0

    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

def tree_depth(tree):
    """
    Calculate the depth of the decision tree.

    Parameters:
    - tree: The decision tree.

    Returns:
    - int: Depth of the tree.
    """
    if not isinstance(tree, dict):
        return 0
    else:
        return 1 + max(tree_depth(subtree) for subtree in tree.values())

def count_nodes_and_leaves(tree):
    """
    Count the number of nodes and leaves in the decision tree.

    Parameters:
    - tree: The decision tree.

    Returns:
    - (int, int): Tuple containing the number of nodes and leaves.
    """
    if not isinstance(tree, dict):
        return 0, 1
    else:
        nodes = 1
        leaves = 0
        for subtree in tree.values():
            subtree_nodes, subtree_leaves = count_nodes_and_leaves(subtree)
            nodes += subtree_nodes
            leaves += subtree_leaves
        return nodes, leaves

# Example of how to use these methods:
# true_labels = [0, 1, 0, 1, 1]
# predicted_labels = [0, 1, 0, 0, 1]
# evaluate_metrics(true_labels, predicted_labels)
# print(confusion_matrix(true_labels, predicted_labels))
# tree = ...  # some decision tree
# print(f"Tree Depth: {tree_depth(tree)}")
# nodes, leaves = count_nodes_and_leaves(tree)
# print(f"Number of Nodes: {nodes}, Number of Leaves: {leaves}")
