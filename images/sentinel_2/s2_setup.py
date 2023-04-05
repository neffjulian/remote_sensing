from s2_download import download_data
from s2_preprocess import preprocess_data

# Sample code for downloading and structuring sentinel data

def create_data(name: str, time_of_interest: str):
    download_data(name, time_of_interest)
    preprocess_data(name)

if __name__ == "__main__":
    create_data("22_march", "2022-03-01/2022-05-31")    
    create_data("22_may", "2022-05-01/2022-05-31")    
    create_data("22_july", "2022-07-01/2022-07-31")