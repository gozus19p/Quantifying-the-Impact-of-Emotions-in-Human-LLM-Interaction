from logic.evaluation.accuracy import AccuracyEvaluation, Confusion


class WordInContextEvaluation(AccuracyEvaluation):
    
    __dataset_identifier: str

    def __init__(self, model: str, version: str, name: str, dataset_identifier: str):
        super().__init__(model=model, version=version, name=name)
        self.__dataset_identifier = dataset_identifier

    
    def get_dataset_identifier(self):
        return self.__dataset_identifier


    def detect(self, result: str, expected: str) -> Confusion:
        expected: bool = expected.lower() == "true"
        result: bool = result.lower() == "true"

        if expected == result == True:
            return Confusion.TP
        if expected == result == False:
            return Confusion.TN
        if expected == False and result == True:
            return Confusion.FP
        return Confusion.FN



