import numpy as np

from sklearn.metrics import roc_auc_score


def predict(model, uid, pids):
    user_latent = model.layers[6].get_weights()[0][uid]
    item_latent = model.layers[2].get_weights()[0][pids]
    
    scores = np.dot(user_latent,item_latent.T)
    
    return scores


def precision_at_k(model, ground_truth, k, user_features=None, item_features=None):
    """
    Measure precision at k for model and ground truth.

    Arguments:
    - lightFM instance model
    - sparse matrix ground_truth (no_users, no_items)
    - int k

    Returns:
    - float precision@k
    """

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    precisions = []

    for user_id, row in enumerate(ground_truth):
        uid_array = np.empty(no_items, dtype=np.int32)
        uid_array.fill(user_id)
        predictions = model.predict(uid_array, pid_array,
                                    user_features=user_features,
                                    item_features=item_features,
                                    num_threads=4)

        top_k = set(np.argsort(-predictions)[:k])
        true_pids = set(row.indices[row.data == 1])

        if true_pids:
            precisions.append(len(top_k & true_pids) / float(k))

    return sum(precisions) / len(precisions)


def full_auc(model, ground_truth):
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):

        predictions = predict(model, user_id, pid_array)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return sum(scores) / len(scores)
