# Live Dream
Augmented reality stream video processing utilizing Google's TensorFlow + DeepDream
<sub>*Note*: This code is being actively developed. We apologize for any bugs, and thanks for reporting any issues!<sub>

### Setup
Clone this repo to your local machine, and add the RNN-Tutorial directory as a system variable to your `~/.profile`. Instructions given for bash shell:

```bash
git clone https://github.com/mrubash1/DeepDream_Streaming_Video
cd DeepDream_Streaming_Video
echo "export DD_STREAM=${PWD}" >> ~/.profile
echo "export PYTHONPATH=${PWD}/src:${PYTHONPATH}" >> ~/.profile
source ~/.profile
```

Create a Conda environment (You will need to [Install Conda](https://conda.io/docs/install/quick.html) first)

```bash
conda create --name tf-dd-stream python=3.5
source activate tf-dd-stream
cd $DD_STREAM
pip install -r requirements.txt
```


### Install TensorFlow
If you have a NVIDIA GPU with [CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) already installed:

```bash
pip install tensorflow-gpu==1.0.1
```
<sub>[Guide for installing CUDA and cuDNN on Linux](https://www.tensorflow.org/versions/r0.11/get_started/os_setup#optional-linux-enable-gpu-support)<sub>

If you will be running TensorFlow on CPU only (e.g. a MacBook Pro), use the following command (if you get an error the first time you run this command read below):
```bash
pip install --upgrade\
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
```
<sub>**Error note** (if you did not get an error skip this paragraph): Depending on how you installed pip and/or conda, we've seen different outcomes. If you get an error the first time, rerunning it may incorrectly show that it installs without error. Try running with `pip install --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl --ignore-installed`. The `--ignore-installed` flag tells it to reinstall the package. If that still doesn't work, please open an [issue](https://github.com/silicon-valley-data-science/RNN-Tutorial/issues), or you can try to follow the advice [here](https://www.tensorflow.org/install/install_mac).</sub>


### Download inception to models folder
```
cd $DD_STREAM
. build/download_inception.sh
```

### Run unittests
We have included example unittests for the `tf_train_ctc.py` script

```bash
# TODO: implement some tests
```


### Run Streaming Deep Dream Demo
All configurations for the RNN training script can be found in `$DD_STREAM/configs/neural_network.ini`

```bash
# TODO: implement the demo
```

_NOTE: If you have a GPU available, the code will run faster if you set `tf_device = /gpu:0` in `configs/neural_network.ini`_


### Add data
TODO: Include Data

    - No data currently available

If you would like to train a performant model, you can add additional jpg files


### Remove additions

We made a few additions to your `.profile` -- remove those additions if you want, or if you want to keep the system variables, add it to your `.bash_profile` by running:

```bash
echo "source ~/.profile" >> .bash_profile
```
