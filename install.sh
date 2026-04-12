#!/bin/bash
pip install -r requirements.txt
pip install darts==0.42.1 --no-deps
pip install tensorflow==2.12.0 --no-deps

# Fix mxnet compatibility with numpy 1.26.4
MXNET_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")/mxnet/numpy/utils.py
sed -i 's/onp\.bool/bool/g' "$MXNET_PATH"
sed -i 's/^bool_ = bool_$/bool_ = np.bool_/' "$MXNET_PATH"
sed -i 's/bool_ = np\.bool_/bool_ = onp.bool_/' "$MXNET_PATH"

python3 -c "import mxnet; print('mxnet ok')"
