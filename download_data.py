import os
import urllib.request
import gzip
from sklearn.datasets import fetch_openml

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Breast Cancer Wisconsin
    wdbc_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    wdbc_path = os.path.join(data_dir, "wdbc.data")
    if not os.path.exists(wdbc_path):
        print("==> Downloading Breast Cancer dataset...")
        urllib.request.urlretrieve(wdbc_url, wdbc_path)
        print(f"    Saved to {wdbc_path}")
        
    # 2. Adult Income
    adult_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    adult_test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    adult_data_path = os.path.join(data_dir, "adult.data")
    adult_test_path = os.path.join(data_dir, "adult.test")
    
    if not os.path.exists(adult_data_path):
        print("==> Downloading Adult Income training dataset...")
        urllib.request.urlretrieve(adult_data_url, adult_data_path)
        print(f"    Saved to {adult_data_path}")
    if not os.path.exists(adult_test_path):
        print("==> Downloading Adult Income testing dataset...")
        urllib.request.urlretrieve(adult_test_url, adult_test_path)
        print(f"    Saved to {adult_test_path}")
        
    # 3. Covertype
    cov_path = os.path.join(data_dir, "covtype.data")
    if not os.path.exists(cov_path):
        print("==> Downloading Covertype dataset (approx. 11MB compressed)...")
        cov_gz_path = os.path.join(data_dir, "covtype.data.gz")
        cov_gz_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        urllib.request.urlretrieve(cov_gz_url, cov_gz_path)
        print("==> Extracting Covertype dataset...")
        with gzip.open(cov_gz_path, 'rb') as f_in:
            with open(cov_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(cov_gz_path)
        print(f"    Extracted and saved to {cov_path}")
        
    # 4. Trigger fetch_openml for MNIST to ensure local sklearn cache is populated
    print("==> Populating MNIST sklearn cache...")
    fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    print("    MNIST cache populated.")
    
    print("\nAll datasets are downloaded and ready in ./data/")

if __name__ == "__main__":
    main()
