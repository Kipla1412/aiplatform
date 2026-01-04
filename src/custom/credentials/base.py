from abc import ABC, abstractmethod

class CredentialProvider(ABC):
    @abstractmethod
    def get_credentials(self) -> dict:
        
        raise NotImplementedError("Subclasses must implement get_credentials")