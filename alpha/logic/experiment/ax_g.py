from logic.experiment.cb import CBExperiment


class AXGExperiment(CBExperiment):

    def get_dataset_identifier(self):
        return "AX-g/sampled.jsonl"
