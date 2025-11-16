class NvidiaSMIError(Exception):
    """A custom exception for validating nvidia-smi availability."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
