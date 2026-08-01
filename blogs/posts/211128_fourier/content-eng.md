title: Why Positional Encoding Makes NeRF more Powerful
date: November 28, 2021
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

### <span id="tl-dr"></span>TL;DR

In this article, we explore why positional encoding increases NeRF's high-fidelity reconstruction ability via exploring the paper: <em>Fourier Features Let Networks Learn High-Frequency Functions in Low-Dimensional Domains </em>

By leveraging a Neural Tangent Kernel (NTK) theory, the authors demonstrate that Fourier features improve the convergence and performance of neural networks on these complex tasks.

## <span id="intro"></span>1. Introduction

Fourier-featuring is a function that embeds a coordinate space point into frequency space.

A prominent example in deep learning is **&#39;Positional Encoding&#39;**, which uses sinusoidal functions to embed coordinate space into frequency space, thereby incorporating positional information that Networks cannot capture.

<figure>
    <img src="./211128_fourier/assets/main.png" alt="Gaussian RT" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"> <strong>Figure 1.</strong> Coordinate-Based MLPs </figcaption>
</figure>

Building upon **NTK theory**, this article foucuses on the theoretical investigation of how neural networks process coordinate information through Fourier-featuring,
    especially for the coordinate-based MLPs, which map dense, continuous low-dimensional input to the high-dimensional output (*e.g.,* NeRF).

## <span id="sec2"></span>2. Background

### <span id="sec2.1"></span>2.1. Kernel Trick

<figure>
    <img src="./211128_fourier/assets/kernel_trick.png" alt="Gaussian RT" width="80%">
    <figcaption style="text-align: center; font-size: 15px;"> <strong>Figure 2.</strong> illustration of Kernel-Trick </figcaption>
</figure>

For a linearly inseparable data point $x$, let $\phi (x)$ be a non-linear mapping function that makes $\phi (x)$ linearly separable.

The kernel trick performs kernel regression without explicitly finding the feature map by defining the kernel as follows:

$$K(x, \ x&#39;) = \phi(x) ^T \phi(x&#39;)
$$

This approach is interpreted as using a feature map $\phi$ with desirable properties through the kernel, rather than mapping input $x$ and then taking the inner product.

### <span id="sec2.2"></span>2.2. Neural Tangent Kernel

**_Neural Tangent Kernel (NTK)_** theory describes the gradient descent-based training of deep neural networks with infinite width through kernel regression, aiming to explain neural networks using the kernel trick.

### <span id="2-2-1-linearization-of-nn-training-kernel-definition"></span>Linearization of NN Training &amp; Kernel

A neural network can be represented by the linearization:

<div class="math-container">$$f(w, \ x) \simeq f(w_0 , \ x) \ + \ \nabla _w f( w_0 , \ x) ^T (w - w_0 )
$$</div>

This Taylor expansion has the following properties:

<ol>
<li>It is <strong><em>linear</em></strong> with respect to the weights $w$.</li>
<li>It is <strong><em>non-linear</em></strong> with respect to $x$.</li>
</ol>

The gradient term $\nabla _w f( w_0 , \ x) ^T (w - w_0 )$ acts as a feature map that maps a non-linear data point $x$ to a useful space.

The corresponding kernel $K$ is defined as follows:

<div class="math-container">
    $$K(x, \ x') =  h_\text{NTK} = \{ \phi(x) , \ \phi (x') \} = \nabla _w f(w_0 , \ x) ^T \ \nabla _w f(w_0, \ x' )
    $$
</div>

### <span id="2-2-2-gradient-based-training-kernel-regression"></span>Gradient-Based Training &amp; Kernel Regression

The NTK can be found through gradient descent in the neural network. For a timestep $t$, gradient descent is expressed as:

<div class="math-container">
    $$w(t+1) \ = \ w(t) - \eta \nabla _w l. $$
</div>

Subsequently, this can be derived as follows:

<div class="math-container">
    $$ { w(t+1)  \ - \ w(t) \over \eta } = -\nabla _w l \simeq {dw \over dt}.$$
</div>

With least squares (MSE) as the loss function,

<div class="math-container">
    $$l(w) = {1 \over 2}  \| f(w, x ) - y \|^2,$$
</div>

the gradient term $\nabla l$ with respect to the $w$ can be derived as

<div class="math-container">
    $$  \nabla _w l= \nabla _w  |f(w, x) - y |. $$
</div>

Therefore, Neural network training via optimization can be represented by NTK kernel regression:

<div class="math-container">
    $$\begin{aligned}
    {d \over dt } y(w) &= \nabla _w f(w, x) ^T \cdot {d \over dt }w \\
    &= - \nabla _w f(w, x) ^T \cdot \nabla _w f(w, x) (f(w, x) - y) \\ &= -h_{\text{NTK}} (f(w,x) -y )
    \end{aligned}$$
</div>

Let $u=y(w)-y$, then the output residual at training iteration $t$ can be written as:

<div class="math-container">$$ u(t) = u(0) \exp (-\eta h_{\text{NTK}} t )
$$</div>

### <span id="sec2.3"></span>2.3. Spectral Bias of DNNs

Based on the NTK approximation, the network&#39;s prediction after $t$ iterations for test data $\mathbf X_\text{test}$ is:

<div class="math-container">
    $$ \hat{\mathbf{y}}^{(t)} \simeq \mathbf{K}_{\text{test}} \mathbf{K}^{-1} ( \mathbf{I} - e^{-\eta \mathbf{K} t} ) \mathbf{y}$$
</div>

For ideal training, $\mathbf K_\text{test} =  \mathbf K$. *i.e.,* equivalent to the last equation in 2.2.2.

By eigendecomposing $\mathbf K = \mathbf Q \mathbf \Lambda \mathbf Q^{\rm T}$, we obtain:

<div class="math-container">
    $$\begin{aligned}
    \mathbf{Q}^{\rm T} (\hat{\mathbf{y}}^{(t)}
    - \mathbf{y}) &\simeq \mathbf{Q}^{\rm T} ( \mathbf{K}_{\text{test}} \mathbf{K}^{-1} ( \mathbf{I} - e ^{-\eta \mathbf{K} t} ) \mathbf{y} - \mathbf{y} )) \\ 
    & \simeq \mathbf{Q}^{\rm T} (  ( \mathbf{I} - e ^{-\eta \mathbf{K} t} ) \mathbf{y} - \mathbf{y} )) \\ 
    & \simeq - e ^{-\eta \mathbf{\Lambda} t}   \mathbf{Q}^{\rm T} \mathbf y \quad (\because e ^{-\eta \mathbf{K} t} = \mathbf{Q} e ^{-\eta \mathbf{\Lambda} t} \mathbf{Q}^{\rm T} ) 
    \end{aligned}$$
</div>

In the above Equation, the exponential decay term decreases with the eigenvalue. It means *larger eigenvalues are learned first.*

For example, in case of the image, large eigenvalues (in spectral domain) correspond to contours, so convergence to high-frequency components is slow without embedding in NeRF.

## <span id="sec3"></span>3. Fourier Features for a Tunable Stationary Neural Tangent Kernel

This section explores how Fourier Features embedding in the kernel space can address convergence issues for high-frequency components.

### <span id="sec3.1"></span>3.1. Fourier-Featuring

The Fourier-Feature mapping function $\gamma$ is defined as:

<div class="math-container">$$\gamma (v) \  = \ \big [a_1 \cos (2 \pi b_1 ^T v), \dots , a_m \cos (2 \pi b_m ^T v), \ a_m \sin (2 \pi b_m ^T v ) \big ]^T
$$</div>
<ul>
<li><strong>Positional Encoding in Transformers:</strong> Adds spatial information to features in attention-based architectures, defined as:
$a_i =1, \ b_i = 10000^{i / d} , \ d : \text{dimension}$</li>
<li><strong>Positional Encoding in NeRF:</strong> Provides even distribution of low & high-frequency information in the input, defined as:
$a_i =1, \ b_i = 2^{i} {}$</li>
</ul>

The kernel induced by this mapping function is:

<div class="math-container">
    $$\begin{aligned}
    K (\gamma (v_1 ) , \  \gamma (v_2) ) &= \gamma (v_1 ) ^T \gamma (v_2) \\ &= \sum _{j=1}^m a^2 _j \cos (2 \pi b_j ^T (v_1 -v_2) ) = h_\gamma (v_1 - v_2 )
    \end{aligned}$$
</div>
<ul>
<li>remember: $\cos (\alpha - \beta ) = \cos \alpha \cos \beta  \ + \ \sin \alpha \sin \beta$</li>
</ul>

This Fourier-feature kernel is a stationary function, meaning it is translation-invariant:

<div class="math-container">$$h_\gamma( (v_1 +k )  -  (v_2 +k ) ) = h_\gamma (v_1  -  v_2 )
    $$
</div>

Coordinate-based MLPs use dense and uniform coordinate points as input. These must be *isotropic* to ensure global performance, meaning features should be extracted in all directions, not just specific ones.

This is why stationary properties that are location-invariant can improve performance. Positional encoding treats all equally distant relations from the coordinate system uniformly, enabling effective high-dimensional space reconstruction.

### <span id="sec3.2"></span>3.2. NTK Kernel with Fourier-Featuring

The NTK Fourier-featured kernel is:

$$K( \phi \circ \gamma (x) , \ \phi \circ \gamma (y) )
$$

Stationary kernel regression here equates to **_convolutional filtering with reconstruction_**, as the neural network approximates the convolution between synthetic kernels $K_\text{NTK}$ and $K_\gamma$ on data points $v_i$ and weights $w_i$.

Thus, the Fourier feature represented by NTK theory is:

<div class="math-container">
    $$f = ( h_\text{NTK} \circ h_\gamma ) * \sum_{i=1}^n w_i \delta _{v_i}$$
</div>

where $\delta$ represents the direction delta.

This expression indicates:

<ol>
<li>A stationary filter $h_\gamma$ extracts information in a <strong><em>location-invariant</em></strong> manner.</li>
<li>Convolution, being the inverse Fourier transform of multiplication in frequency space, allows extraction of features across different frequencies in a multifaceted (yet location-invariant) way through components of specific frequencies directly embedded in $h_\gamma$.</li>
<li>A Neural Network, receiving Fourier-featured input, is equivalent to performing kernel regression by combining NTK and a stationary kernel.</strong></li>
</ol>
<video controls style="width: 100%;"><source src="https://bmild.github.io/fourfeat/img/lion_none_gauss_v1.mp4" type="video/mp4"></video>
