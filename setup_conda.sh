##Conda version
conda version 22.11.1
/homes/ac.rgnanaolivu/miniconda3/bin/conda


# version of python to download
PYTHON=3.9.15

#step 1
conda create -n drugcell_python python=3.9.15 anaconda


#step 2
conda activate drucell_python


#step 3
conda env update --name drugcell_python --file environment.yml

#step 4
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 torchmetrics==0.11.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/ECP-CANDLE/candle_lib@0d32c6bb97ace0370074194943dbeaf9019e6503
