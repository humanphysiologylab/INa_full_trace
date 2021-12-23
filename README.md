# INa_full_trace

How to create conda environment:
```sh
conda env create -f environment.yml --prefix ./env
conda activate ./env
conda deactivate  # if you need
```

How to remove the environment:
```sh
conda remove --prefix ./env --all
```
How to reconstuct file.so
```
cd src/model_ctypes/M1/
make clean && make ina && make
```
