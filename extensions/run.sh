rm -rf build

pip uninstall brainpylib -y

python setup_cuda.py bdist_wheel

pip install dist/brainpylib-0.0.6-cp38-cp38-linux_x86_64.whl

mv /home/adadu/miniconda3/envs/py38/lib/python3.8/site-packages/brainpylib/gpu_ops.cpython-38-x86_64-linux-gnu.so /home/adadu/miniconda3/envs/py38/lib/python3.8/site-packages/brainpylib/gpu_ops.so

cd ./tests/

python test_vmmm.py

