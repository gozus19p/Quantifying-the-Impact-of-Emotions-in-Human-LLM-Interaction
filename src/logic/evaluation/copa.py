from logic.evaluation.accuracy import AccuracyEvaluation, Confusion


class COPAEvaluation(AccuracyEvaluation):
    
    __dataset_identifier: str

    def __init__(self, model: str, version: str, name: str, dataset_identifier: str):
        super().__init__(model=model, version=version, name=name)
        self.__dataset_identifier = dataset_identifier

    
    def get_dataset_identifier(self) -> str:
        return self.__dataset_identifier


    def detect(self, result: str, expected: str) -> Confusion:
        if expected == result == "1":
            return Confusion.TP
        if expected == result == "0":
            return Confusion.TN
        if expected == "1" and result == "0":
            return Confusion.FP
        return Confusion.FN


