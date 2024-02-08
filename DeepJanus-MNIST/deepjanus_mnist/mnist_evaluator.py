import os
from typing import TYPE_CHECKING

import numpy as np
from deepjanus.evaluator import Evaluator
from deepjanus.log import get_logger

from .mnist_config import MNISTConfig

if TYPE_CHECKING:
    from .mnist_member import MNISTMember

log = get_logger(__file__)


class MNISTLocalEvaluator(Evaluator):
    """Executes a local ??? instance and uses it to evaluate members."""

    def __init__(self, config: MNISTConfig):
        self.config = config

        self.model_file = str(config.FOLDERS.models.joinpath(config.MODEL_FILE))
        if not os.path.exists(self.model_file):
            raise Exception(f'File {self.model_file} does not exist!')

        self.model = None

    def evaluate(self, member: 'MNISTMember', max_attempts=20) -> bool:
        # TODO: mbr evaluation
        if self.model is None:
            import tf_keras.models
            self.model = tf_keras.models.load_model(self.model_file)

        batch = np.array([member.bitmap]).reshape((-1,28,28,1))
        # Array containing the confidence for each label (digit 0-9)
        confidences = self.model.predict(batch)[0]

        best_label, second_best_label = np.argsort(confidences)[:2]

        confidence_expected_label = confidences[member.expected_label]
        if best_label == member.expected_label:
            confidence_other_best = confidences[second_best_label]
        else:
            confidence_other_best = confidences[best_label]

        prediction_quality = confidence_expected_label - confidence_other_best

        # Requirement: digit should be classified correctly
        satisfy_requirements = best_label == member.expected_label

        # Update member here to ensure that log contains evaluation info
        member.satisfy_requirements = satisfy_requirements
        member.predicted_label = best_label
        member.prediction_quality = prediction_quality
        log.info(f'{member} evaluation completed')

        return satisfy_requirements
