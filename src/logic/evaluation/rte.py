from logic.evaluation.accuracy import AccuracyEvaluation, Confusion


class RTEEvaluation(AccuracyEvaluation):
    
    __dataset_identifier: str

    def __init__(self, model: str, version: str, name: str, dataset_identifier: str):
        super().__init__(model=model, version=version, name=name)
        self.__dataset_identifier = dataset_identifier

    
    def get_dataset_identifier(self):
        return self.__dataset_identifier


    def detect(self, result: str, expected: str) -> Confusion:
        if expected == result == "entailment":
            return Confusion.TP
        if expected == result == "not_entailment":
            return Confusion.TN
        if expected == "not_entailment" and result == "entailment":
            return Confusion.FP
        return Confusion.FN


