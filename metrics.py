# METRICS PART

def get_fp(confusion_matrix, label):
    sum=0
    for key in confusion_matrix.keys():
        if((f"t-{label}" not in key) and (f"-p-{label}" in key)):
            sum+=confusion_matrix[key]
    return sum
def get_tn(confusion_matrix, label):
    sum=0
    for key in confusion_matrix.keys():
        if((f"t-{label}" not in key) and (f"-p-{label}" not in key)):
            sum+=confusion_matrix[key]
    return sum

def get_fn(confusion_matrix, label):
    sum=0
    for key in confusion_matrix.keys():
        if((f"t-{label}" in key) and (f"-p-{label}" not in key)):
            sum+=confusion_matrix[key]
    return sum
    

def evaluate_metrics(true_labels, predicted_labels, labels):
    """
    Evaluate and print various metrics for the classification.

    Parameters:
    - true_labels: list or array of true class labels.
    - predicted_labels: list or array of predicted class labels.
    - labels list on all possible classes

    """

    matrix={}
    for true_label in labels:
        for predicted_label in labels:
            matrix["t-"+true_label+"-p-"+predicted_label]=0

    for true, pred in zip(true_labels, predicted_labels):
        if(pred!=None):
            matrix["t-"+true+"-p-"+pred]+=1

    print("")
    for label in labels:
        tp = matrix["t-"+label+"-p-"+label]
        fp = get_fp(matrix, label)
        tn = get_tn(matrix, label)
        fn = get_fn(matrix,label)
        accuracy = (tp + tn) / (tp+tn+fp+fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print(f"Metrics for class {label}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")
        print("")

def confusion_matrix(true_labels, predicted_labels, labels):
    """
    Compute the confusion matrix.

    Parameters:
    - true_labels: list or array of true class labels.
    - predicted_labels: list or array of predicted class labels.
    - labels list on all possible classes

    """
    matrix={}
    for true_label in labels:
        for predicted_label in labels:
            matrix["t-"+true_label+"-p-"+predicted_label]=0

    for true, pred in zip(true_labels, predicted_labels):
        if(pred!=None):
            matrix["t-"+true+"-p-"+pred]+=1
    
    print("")
    print("Confusion Matrix")
    print("")
    count=0
    for value in matrix.values():
        if((count+1)%len(labels)==0):
            print(value)
        else:
            print(value, end=" ")
        count+=1
    print("")
    print("Y-axis: True labels")
    for label in labels:
        print(label, end=" ")
    print("")
    print("")
    print("X-axis: Predicted labels")
    for label in labels:
        print(label, end=" ")
    print("")
    print("")

    return matrix

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
