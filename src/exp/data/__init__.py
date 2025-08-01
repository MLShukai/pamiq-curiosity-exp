from enum import StrEnum, auto


class BufferName(StrEnum):
    """Enumerates all buffer names in the experiments."""

    IMAGE = auto()
    FORWARD_DYNAMICS = auto()
    POLICY = auto()


class DataKey(StrEnum):
    """Enumerates all data key names in the experiments."""

    OBSERVATION = auto()
    HIDDEN = auto()
    ENCODER_HIDDEN = auto()
    PREDICTOR_HIDDEN = auto()
    ACTION = auto()
    ACTION_LOG_PROB = auto()
    REWARD = auto()
    VALUE = auto()
    TARGET = auto()
    UPPER_ACTION = auto()
