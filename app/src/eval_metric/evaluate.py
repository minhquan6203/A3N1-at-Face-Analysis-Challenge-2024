from sklearn.metrics import accuracy_score, f1_score

class ScoreCalculator:
    def acc(self, labels, preds) -> float:
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        return accuracy_score(labels, preds)

    def f1(self, labels, preds) -> float:
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        return f1_score(labels, preds, average='macro')