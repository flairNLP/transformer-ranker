import pytest
import torch
from datasets import load_dataset


@pytest.fixture(scope="session")
def iris_dataset():
    """Prepares a sample classification dataset."""
    dataset = load_dataset("skorkmaz88/iris", split="train", trust_remote_code=True)
    features = torch.tensor(
        [[row["sepal.length"], row["sepal.width"], row["petal.length"], row["petal.width"]]
         for row in dataset], dtype=torch.float32)
    labels = torch.tensor(dataset["variety"], dtype=torch.float32)

    return {"data": features, "labels": labels}


@pytest.fixture(scope="session")
def california_housing_dataset():
    """Prepares a sample regression dataset."""
    dataset = load_dataset("gvlassis/california_housing", split='train')
    features = torch.tensor(
        [[row['MedInc'], row['HouseAge'], row['AveRooms'], row['AveBedrms'],
          row['Population'], row['AveOccup'], row['Latitude'], row['Longitude']]
         for row in dataset], dtype=torch.float32)
    labels = torch.tensor(dataset['MedHouseVal'], dtype=torch.float32)

    return {"data": features, "labels": labels}
