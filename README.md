# Cr-Fasteid
Cr-Fastreid (Compression and Reconstruction Fastreid) is a project that stores the image data in the float format and applies the re-identification on the data. Reid has been increasingly important and applied in many areas like people re-identification, car re-identification and etc. However, due to the cost of storage and the vulnurability of image data, it's often difficult to store the full-size image data and it has brought much difficulty due to the demaged image data. With CrFastreid, JD-Fastreid can now be applied to the image data with float format, and it saves much storage cost and makes it easier to restore the demaged data.
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
