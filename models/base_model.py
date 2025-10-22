"""Base model interface."""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def compile(self):
        raise NotImplementedError
