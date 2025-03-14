import pytest
import torch
from datasets import load_dataset


@pytest.fixture(scope="session")
def iris_dataset():
    """Prepares a sample classification dataset."""
    dataset = load_dataset("skorkmaz88/iris", split="train", trust_remote_code=True)
    features = torch.tensor(
        [[row["sepal.length"], row["sepal.width"], row["petal.length"], row["petal.width"]] for row in dataset],
        dtype=torch.float32,
    )  # fmt: skip
    labels = torch.tensor(dataset["variety"], dtype=torch.float32)

    return {"data": features, "labels": labels}


@pytest.fixture(scope="session")
def california_housing_dataset():
    """Prepares a sample regression dataset."""
    dataset = load_dataset("gvlassis/california_housing", split="train", trust_remote_code=True)
    features = torch.tensor(
        [[row["MedInc"], row["HouseAge"], row["AveRooms"], row["AveBedrms"]] for row in dataset],
        dtype=torch.float32,
    )  # fmt: skip
    labels = torch.tensor(dataset["MedHouseVal"], dtype=torch.float32)

    return {"data": features, "labels": labels}


def generate_sample_dataset(k: int = 6, dim: int = 1024, distance: float = 1.0, radius: float = 0.3):
    """Generates a synthetic dataset for testing knn."""
    num_correct = (k // 2) + 2  # +1 for majority, +1 for the datapoint itself
    num_incorrect = (k - 1) // 2  # these will be the minority
    num_total = 2 * (num_correct + num_incorrect)

    # Generate a three-class dataset with the datapoints on two spheres
    data = torch.nn.functional.normalize(torch.rand(num_total, dim), dim=1) * radius
    labels = torch.tensor([0] * num_correct + [2] * num_incorrect + [1] * num_correct + [2] * num_incorrect)
    diff = torch.nn.functional.normalize(torch.rand(dim), dim=0) * distance

    data += torch.rand(dim)
    data[num_correct + num_incorrect :] += diff
    expected_knn_accuracy = num_correct / (num_correct + num_incorrect)
    return data, labels, expected_knn_accuracy
