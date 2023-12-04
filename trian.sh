pip install -r LRRU/requirements.txt
pip3 install opencv-python
pip3 install opencv-python-headless
python setup.py build_ext --inplace

wandb login --cloud 4b93c33759802a756c0a5a1b6059dd6d8f3b93ee
python LRRU/train_apex.py 
