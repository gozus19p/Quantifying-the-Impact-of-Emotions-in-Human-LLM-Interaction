import pandas as pd

class Dataset:
    __dataframe: pd.DataFrame
    __name: str
    __url: str
    
    def __init__(self, dataframe: pd.DataFrame, name: str, url: str):
        assert dataframe is not None, "Dataframe cannot be None"
        assert name is not None, "Name cannot be None"
        assert url is not None, "URL cannot be None"
        self.__dataframe = dataframe
        self.__name = name
        self.__url = url
    
    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__dataframe
    
    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def url(self) -> str:
        return self.__url
    
