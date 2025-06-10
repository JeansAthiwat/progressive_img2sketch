# Good Kid M.a.a.D city is a good album

# Current Pipeline rough idea

We start from an RGB camera image of a single building
 |
\/
Extract the line drawing sketch (Informative Drawings looks like a good model to use)
 |
\/
extract the perspective lines and basic geometric shapes (such as rectangles, triangles, and circles). I believe this should be the first layer of progressive line details, helping the user understand the building’s overall structure in a simplified, abstract form. (not sure if this step is needed if the LOD model is already good)
 |
\/
find some LOD model/algorithm to extract the line drawing sketch
 |
\/
create an iterative interface that can gradually fill in the details from primitive shape and then iterate through the extracted each LOD level

# Getting started with TripoSR

```conda create -n athiwat_TripoSR  python=3.10```

```conda activate athiwat_TripoSR```
```pip3 install torch torchvision torchaudio```

```python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"```
```conda install -c nvidia cuda-toolkit=12.6```
```conda install -c conda-forge gcc gxx```
```pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472```
```pip install --upgrade setuptools```





```pip install -r requirements.txt```