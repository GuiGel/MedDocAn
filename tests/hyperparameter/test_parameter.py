from meddocan.hyperparameter.parameter import Parameter, ParameterName


class TestParameterName:
    def test_init(self):
        for name in ParameterName.__members__.keys():
            Parameter(name)
