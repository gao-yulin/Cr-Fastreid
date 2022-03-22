# naic2022semi-final
## Installation
```
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r docs/requirements.txt
```
## Inference
```shell
# extract the picture feature
python project/extract.py
# run simple demo
python exmple.py
# apply Reid inference
python project/reid.py
# results saved in reid_results/*.json
```