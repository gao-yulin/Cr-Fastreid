# naic2022semi-final
## Installation
```
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r docs/requirements.txt
```
## Submission format
### Input data format
Default path: 
```
# query data
"./datasets/NAIC2021Reid/query_feature A"
# gallery data
"./datasets/NAIC2021Reid/gallery_feature A"
```
Desired path:
```
# query data
"./reconstructed_query_feature/64"
# gallery data
"./datasets/NAIC2021Reid/gallery_feature A"
```
## Set dataset path
```shell
export FASTREID_DATASETS=project/datasets/
```
## Training:
```
 # from scratch
 python project/tools/train_net.py --config-file project/configs/NAIC2021Reid/sbs_mlp2x.yml MODEL.DEVICE "cuda:0"
 # resume from previous model
 python project/tools/train_net.py --config-file project/configs/NAIC2021Reid/sbs_mlp2x.yml --resume MODEL.DEVICE "cuda:0"
 # Outputs at project/logs/NAIC2021Reid/sbs_mlp2x
 python project/tools/train_net.py --config-file project/configs/NAIC2021Reid/r34-ibn.yml MODEL.DEVICE "cuda:0"
 python project/tools/train_net.py --config-file project/configs/NAIC2021Reid/nest101-base.yml MODEL.DEVICE "cuda:0"
 
```
## Inference
```
python project/tools/train_net.py --config-file project/configs/NAIC2021Reid/sbs_mlp2x_inference.yml --infer-only MODEL.DEVICE "cuda:0"
# Submit file at project/logs/NAIC2021Reid/sbs_mlp2x_inference/Inference_On_NAIC2021ReidTestA/results.json

