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
# apply Reid inference
python project/reid.py
# results saved in reid_results/*.json
```