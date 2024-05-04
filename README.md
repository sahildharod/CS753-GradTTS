## GRAD-TTS : Diffusion Probabilistic Model for Text-to-Speech

Team Members : Sahil Dharod (210070026), Azeem Motiwala (210070018), Jay Chaudhary (210070022), Shlesh Gholap (210070080)

The paper GRAD-TTS presented the first acoustic feature generator utilizing the concept of diffusion probabilistic modelling. The main generative engine of Grad-TTS is the diffusion-based decoder that transforms Gaussian noise parameterized with the encoder output into mel-spectrogram while alignment is performed with Monotonic Alignment Search. The model we propose allows to vary the number of decoder steps at inference, thus providing a tool to control the trade-off between inference speed and synthesized speech quality.

In this hacker role, we made the following changes to the original implementation:
1) Drawing inspiration from lightweight models, we replaced the regular convolutions in the ResNet block of the decoder which has a UNet architecture with depthwise separable convolutions to reduce parameters and computation
2) One of the limitations/future work proposed by the authors was to try any other variance schedule apart from 'linear'
   We have implemented cosine noise scheduling for the diffusion process which is given by :
   
   $min (1 - \frac{\alpha_t}{\alpha_{t-1}},0.999), \alpha_t = \frac{f(t)}{f(0)}$ where $f(t) = cos^2(\frac{i + 0.008}{1 + 0.008}*\frac{\pi}{2})$ and $i = \frac{t - 1}{T - 1}$

### Installation
Firstly install the Python package requirements according to the original implementation (Preferably in a docker container as pytorch version 1.9.0 isn't compatible with python 3.8 or higher in an environment and the original code wass tested on python 3.6.9)
```bash
pip install -r requirements.txt
```
To resolve other version related errors, install the following packages:
```bash
pip install torchaudio==0.9.0
pip install setuptools==59.5.0
pip install numba
```
Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```
Download and extract the LJSpeech dataset using the following commands
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```
Due to incompatibility of Cuda(12.2) and pytorch(1.9.0) versions, we could not train using the command given by the authors on GPU, use the following command to perform training on cpu
```bash
CUDA_VISIBLE_DEVICES= python train.py
```

