# LOLCAT
This is the repository for LOLCAT - Local Latent Concatentation and Attention. LOLCAT aims to decode cell type and class from in vivo spike times from single neurons. The model uses attention to find specific points in time that are meaningful to for differentiating cell types.

You can find more details in our paper:  
>Schneider, A., Azabou, M., McDougall-Vigier, L., Parks, D. B., Ensley, S., Bhaskaran-Nair, K., Nowakowski, T., Dyer, E. L. & Hengen, K. B. (2022). Transcriptomic cell type structures in vivo neuronal activity across multiple time scales. Cell Reports, Volume 42, Issue 4, 2023 [Link](https://www.cell.com/cell-reports/fulltext/S2211-1247(23)00329-7)


![](lolcat_architecture.png)

## Setup
To set up a Python virtual environment with the required dependencies, run:
```bash
python3 -m venv lolcat_env
source lolcat_env/bin/activate
pip install --upgrade pip wheel
pip install scipy absl-py==0.12.0 tensorboard==2.6.0
````

Install [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), it is recommended to follow the instructions for your specific system. We expect the code to work with all recent versions of PyTorch and PyG, including PyTorch 2.0. If you want to use the same versions we used, you can run the following to install PyTorch 1.9.1 and PyG (PyTorch Geometric):
```bash
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

## Applying LOLCAT to your own data
In LOLCAT, the time series of neuronal activity of a single neuron,
is split into short snippets of duration $T$ (we use $T=3s$). 
This collection of snippets is treated as a set, meaning that the order of the snippets does not matter.
These snippets can be collected from contiguous or non-contiguous recordings, and we can collect an 
arbitrary number of them.
We use the PyTorch Geometric (PyG) package, which provides a simple data representation of a set. 

Dataset $\mathcal{D}=\{ (\mathcal{X}_i), y_i \}$ is a collection of neurons, where each neuron $i$ is characterized 
by their set of snippets $\mathcal{X}_i$ and cell type label $y_i$. We process the snippets to extract the Inter-Event Intervals (IEI),
resulting in a feature vector in $\mathbb{R}^D$ for each snippet, or $\mathcal{X}_i = \{ x_i^{(1)}, \cdots, x_i^{(N_i)}\}$, where $N_i$ is the 
number of snippets for neuron $i$, and can be different for different neurons. 

In PyG, a set can be represented by object `data`, which will hold the following attributes:
- `data.x`: Node feature matrix with shape [num_snippets, D]
- `data.y`: An integer scalar corresponding to the cell type.

```python
# Example
import torch
from torch_geometric.data import Data

iei_matrix = torch.rand((12, 90)) # 12 snippets, 90-d iei vector
cell_type = torch.tensor(3, dtype=torch.long) 
data = Data(x=iei_matrix, y=cell_type)
```

Combine all your neurons into a dataset object, we provide a simple class `lolcat.InMemoryDataset` 
which has useful utilities but you can build a dataset object from scratch.
```python
# Example
class CustomDataset(lolcat.InMemoryDataset):
    def process(self):
        data_list = []

        num_neurons = 10
        for i in range(num_neurons):
            # create iei matrix
            num_snippets = torch.randint(20, 200, (1,)).item()
            iei_matrix = torch.rand((num_snippets, 90)) # 12 snippets, 90-d iei vector

            # create cell type
            cell_type = torch.randint(4, (1,)).item()
            data = Data(x=iei_matrix, y=cell_type)
            data_list.append(data)
        return dict(data_list=data_list)
```

Call PyG's dataloader, which will take care of batching sets with different sizes. If you want to learn more about how the batching is done, refer to [documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches).
```python
# Example
from torch_geometric.data import DataLoader
dataset = CustomDataset(root='data/', 'my_dataset')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

LOLCAT is then trained to predict the cell type from **a set of snippets**.
```python
# Example
model = LOLCAT(...)

for data in loader:
    x, batch, target = data.x.to(device), data.batch.to(device), data.y.to(device)
    logits, _ = model(x, batch)
```

## Downloading from the Allen Institute Visual Coding Dataset
If you want to download data from the Allen Institute, you will need to install the AllenSDK, the sdk requires Python 3.8 or lower:
```bash
pip install allensdk
```
then run the following to download the data:
```bash
python download_allensdk.py --root ./data
```

## Code
Thanks for your interest in the project! We are currently working on providing more code usage examples using the visual coding dataset. Please check back for more updates!

--- 

If you have any questions or comments, please feel free to reach out to us. 
