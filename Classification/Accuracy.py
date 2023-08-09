import numpy as np
from sklearn.metrics import top_k_accuracy_score


class KRank:
    def __init__(self, match_scores_matrix, y_test, reference_labels):
        self.match_scores_matrix = match_scores_matrix
        self.y_test = y_test
        self.reference_labels = reference_labels

    def find_k_min_value_labels(self, labels, row, K):
        # return labels associated with the k min values in the row!
        if len(labels) == len(row):
            indices = np.argsort(row)[:K]
            min_value_labels = []
            for index in indices:
                min_value_labels.append(labels[index])
            return min_value_labels
        else:
            raise ValueError('Every score should have a label')


    def get_k_rank_accuracy(self, k=3):
        correct = 0
        incorrect = 0
        for correct_prob_label, current_probe_scores in zip(self.probe_labels, self.match_scores_matrix):
            top_reference_labels = self.find_k_min_value_labels(self.reference_labels, current_probe_scores, k)
            if correct_prob_label in top_reference_labels:
                correct += 1
            else:
                incorrect += 1
        return correct / (correct + incorrect)

    def get_up_to_k_rank_accuracy(self, k=3):
        rank_accuracies = []
        for i in range(1, k+1):
            rank_accuracies.append(top_k_accuracy_score(self.y_test, self.match_scores_matrix, k=i))
        return rank_accuracies