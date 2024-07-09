from abc import ABC, abstractmethod

class baseModel(ABC):
    @abstractmethod
    def fit(self, U):
        pass
    
    @abstractmethod
    def predict(self, xi, sj):
        pass
    
    @abstractmethod
    def recommend(self, xi, n):
        pass