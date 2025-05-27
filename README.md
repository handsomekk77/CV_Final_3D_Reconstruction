## Installation

### 1. Clone Mast3r

```bash
git clone --recursive https://github.com/naver/mast3r
cd mast3r
# if you have already cloned mast3r:
# git submodule update --init --recursive 
```

### 2. Create the environment
```bash
conda create -n mast3r python=3.11 cmake=3.14.0
conda activate mast3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
```
### 3. Download CV_Final_3D_Reconstruction Code
```bash
git clone https://github.com/handsomekk77/CV_Final_3D_Reconstruction.git
```
### 4. Revise the dir in batch_inference.py
```bash
root_dir = r"your path/7SCENES"
output_dir = r"your\Results"
json_path = r"your path\reconstruction_sequences.json"
python_exe = "python"  # or "python3" depending on your system
script_path = r"ypur path\3d_reconstruction.py"  # <-- change to your script name
```

## Checkpoints

### Download checkpoints
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```
## Start Inference
```bash
python batch_inference.py
```




