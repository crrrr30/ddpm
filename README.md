# Deep Denoising Probabilistic Model
## A Custom PyTorch Implementation

This is a custom PyTorch implementation of a DDPM, based partly on HuggingFace's tutorial (https://huggingface.co/blog/annotated-diffusion).

Let $\boldsymbol x_0\sim q(\boldsymbol x_0)$ denotes the data distribution. Define a forward diffusion process as
$$q(\boldsymbol x_{t+1}\mid\boldsymbol x_t)=\mathcal N(\boldsymbol x_{t+1}\mid\boldsymbol x_t,\beta_tI).$$
