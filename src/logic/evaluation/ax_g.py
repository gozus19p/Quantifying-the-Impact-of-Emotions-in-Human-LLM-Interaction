from logic.evaluation.cb import CBEvaluation


class AXGEvaluation(CBEvaluation):

    __dataset_identifier: str

    def __init__(self, model: str, version: str, name: str, dataset_identifier: str):
        super().__init__(model=model, version=version, name=name)
        self.__dataset_identifier = dataset_identifier

    def get_dataset_identifier(self) -> str:
        return self.__dataset_identifier
