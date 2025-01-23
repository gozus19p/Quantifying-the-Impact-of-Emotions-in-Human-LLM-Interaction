import re

from logic.evaluation.accuracy import AccuracyEvaluation, Confusion


class COPAEvaluation(AccuracyEvaluation):
    
    __dataset_identifier: str

    def __init__(self, model: str, version: str, name: str, dataset_identifier: str):
        super().__init__(model=model, version=version, name=name)
        self.__dataset_identifier = dataset_identifier

    
    def get_dataset_identifier(self) -> str:
        return self.__dataset_identifier


    def detect(self, result: str, expected: str) -> Confusion:
        expected_option_1: bool = expected == "0"
        expected_option_2: bool = expected == "1"
        if not expected_option_1 and not expected_option_2:
            raise ValueError(f"Invalid expected value: {expected}")
        
        result_option_1: bool = re.search(r"\boption 1\b", result, re.IGNORECASE)
        result_option_2: bool = re.search(r"\boption 2\b", result, re.IGNORECASE)
        if not result_option_1 and not result_option_2:
            return Confusion.FN
        
        expected_bool: bool = True if expected_option_1 else False
        result_bool: bool = True if result_option_1 else False

        if expected_bool == result_bool == True:
            return Confusion.TP
        
        if expected_bool == result_bool == False:
            return Confusion.TN
        
        if expected_bool == False and result_bool == True:
            return Confusion.FP    
    
        return Confusion.FN


