# Deep Denoising Probabilistic Model
## A Custom PyTorch Implementation

This is a custom PyTorch implementation of a DDPM, based partly on HuggingFace's tutorial (https://huggingface.co/blog/annotated-diffusion).

Let $\boldsymbol x_0\sim q(\boldsymbol x_0)$ denotes the data distribution. Define a $T$-step forward diffusion process as
$$q(\boldsymbol x_t\mid\boldsymbol x_{t-1})=\mathcal N(\boldsymbol x_{t+1};\sqrt{1-\beta_t}\boldsymbol x_t,\beta_tI),$$
with $\boldsymbol x_0$ being true data points and $t\in[\\![1,\cdots,T]\\!]$. We expect that, after the transformation, $x_T\sim\mathcal N(\boldsymbol 0,I)$.

Reparameterization gives
$$q(\boldsymbol x_t\mid\boldsymbol x_0)=\mathcal N(\boldsymbol x_t;\sqrt{\bar\alpha_t}\boldsymbol x_0,(1-\bar\alpha_t)I),$$
where $\bar\alpha_t=\prod_{i=1}^t\alpha_i$ and $\alpha_t=1-\beta_t$.

If $x_T\sim\mathcal N(\boldsymbol 0,I)$, then fitting a reverse transform $p_\theta(\boldsymbol x_t\mid x_{t-1})$ allows us to generate samples starting from Gaussian noise.

Let $p_\theta(\boldsymbol x_{t-1}\mid\boldsymbol  x_t)=\mathcal N(\boldsymbol x_{t-1};\mu_\theta(\boldsymbol x_t,t),\tilde\beta_tI)$ and 
$$\mu_\theta(\boldsymbol x_t,t)=\frac1{\sqrt{\alpha_t}}\left(\boldsymbol x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\cdot\boldsymbol\epsilon_\theta(\boldsymbol x_t,t)\right).$$

$\boldsymbol\epsilon_\theta$ can then be approximated with a neural network that has objective
$$L(\theta)\coloneqq\mathbb E[\Vert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\boldsymbol x_t,t)\Vert^2]=\mathbb E[\Vert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\boldsymbol x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,t)\Vert^2]$$
over $t\sim\mathrm{Uniform}[\\![1,\cdots,T]\\!]$, $\boldsymbol\epsilon\sim\mathcal N(\boldsymbol 0,I)$, and $x_0\sim q(\boldsymbol x_0)$.
