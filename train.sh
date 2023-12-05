pip install -r LRRU/requirements.txt
pip3 install opencv-python
pip3 install opencv-python-headless
python setup.py build_ext --inplace

python LRRU/train_apex.py 
