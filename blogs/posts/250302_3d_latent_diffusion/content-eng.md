title: An Era of 3D Generation: From ShapeVAE to Trellis and Hunyuan 3D
date: March 02, 2025
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

<button id="copyButton">
    <i class="bi bi-share-fill"></i>
</button>

<div id="myshare_modal" class="share_modal">
    <div class="share_modal-content">
        <span class="share_modal_close">×</span>
        <p><strong>Link Copied!</strong></p>
        <div class="copy_indicator-container">
        <div class="copy_indicator" id="share_modalIndicator"></div>
        </div>
    </div>
</div>

<nav class="toc">
    <ul><li><a href="#h2-1">Preliminary: What is Latent?</a><ul></ul></li><li><a href="#h2-2">ShapeVAE</a><ul><li><a href="#h3-1">Challenges for ShapeVAE</a></li></ul></li><li><a href="#h2-3">Trellis</a><ul><li><a href="#h3-2">Structured Latent</a></li><li><a href="#h3-3">3D Generation</a></li></ul></li><li><a href="#h2-4">Hunyuan3D-v2</a><ul><li><a href="#h3-4">Hunyuan-ShapeVAE</a></li><li><a href="#h3-5">Hunyuan3D-DiT</a></li><li><a href="#h3-6">Hunyuan3D-Paint</a></li></ul></li><li><a href="#h2-6"> Trellis vs Hunyuan? </a><ul></ul></li><li><a href="#h2-5">Closing</a><ul></ul></li></ul>
</nav>

<hr/>

<figure id="figure-0" >
  <img src='./250302_3d_latent_diffusion/assets/teaser.gif' alt='img alt' width='100%'>
</figure>

<p id="p-2"><strong id="strong-81" ><em> Diffusion</em></strong>. From Imagen, DALL·E to Stable Diffusion and Midjourney, Diffusion models have surpassed GANs to become the standard paradigm for modern 2D generative models. In particular, the advent of <em id="em-1"><strong id="strong-1">Latent Diffusion</strong></em>, with its promise of:</p>
<p id="p-3"><em id="em-37" >"Applying Diffusion in Latent Space → Reduced Computation + High-Resolution Image Generation"</em></p>
<p id="p-4">has led to the presentation of 2D generative models that achieve high performance and efficiency, enabling speeds and quality suitable for real-world applications.</p>
<p id="p-5">The expansion of generative models extends beyond 2D to the realms of video and 3D. In 2024, video generation models such as Sora, DreamMachine (Ray), and Veo have demonstrated the potential for modality expansion. </p>
<p id="p-6">In recent days, beyond video, latent diffusion-based models are also proving their capabilities in the 3D domain.</p>
<p id="p-7">This article delves into the concept of <em id="em-2"><strong id="strong-2">3D Latent Diffusion</strong></em> and analyzes its core component, <em id="em-3"><strong id="strong-3">ShapeVAE</strong></em>, examining how it overcomes the limitations of traditional Score Distillation Sampling (SDS) and NeRF-based Large Reconstruction Models (LRMs). Furthermore, we will compare and contrast the state-of-the-art 3D generation models, Trellis and Hunyuan3D, providing an in-depth exploration of their design differences, strengths, and weaknesses.</p>
<h2 id="h2-1">Preliminary: What is Latent?</h2>
<table id="table-1">
<thead>
<tr>
<th>RBF Network</th>
<th>Gaussian Mixture Model</th>
<th>3D Gaussian Splatting</th>
</tr>
</thead>
<tbody><tr>
<td><img id="img-1" src="./250302_3d_latent_diffusion/assets/image-8.png" width="200px" /></td>
<td><img id="img-2" src="./250302_3d_latent_diffusion/assets/image-2.png" width="200px" /></td>
<td><img id="img-3" src="./250302_3d_latent_diffusion/assets/image-4.png" width="200px" /></td>
</tr>
</tbody></table>
<p id="p-8">Consider classical machine learning concepts like Radial Basis Function (RBF) networks and Gaussian Mixture Models (GMMs).  The core idea of both is to approximate a data distribution (or function) by combining:</p>
<ul id="ul-2">
<li id="li-2">Several basis functions, and</li>
<li id="li-3">The weight of each basis function.</li>
</ul>
<p id="p-9">$$
f(x) \approx \sum_{i=1}^{N} w_i \phi(||x - c_i||)
$$</p>

<p id="p-10">Equation: Radial Basis Function (RBF)</p>
<p id="p-11">From a similar perspective, 3D Gaussian Splatting can also be interpreted as approximating data (multi-view observations of a 3D scene) through a combination of:</p>
<ul id="ul-3">
<li id="li-4">Multiple basis functions (3D Gaussian primitives), and</li>
<li id="li-5">The weight of each basis function (the opacity of each Gaussian).</li>
</ul>
<p id="p-12">
$$
\text{Scene}(x) \approx \sum_{i=1}^{N} \alpha_i  G_i(x; \mu_i, \Sigma_i)
$$</p>
<p id="p-13">Equation: 3D Gaussian Splatting</p>
<p id="p-14">Here, the RBF and Gaussian primitives each have learnable parameters (e.g., mean, variance) that are optimized during the learning process.</p>
<p id="p-15">Shifting our perspective slightly, these basis functions (primitives) can be considered a type of <em id="em-4">'latent vector'</em> that compresses the meaning of the data distribution.  The entire collection can be viewed as a <em id="em-5">'latent vector set'</em>.</p>
<p id="p-16">RBF networks, 3D Gaussian Splatting, etc., define their basis representations in a human-crafted manner. However, if we learn the basis functions in a learnable way for a given data distribution, this becomes <em id="em-6"><strong id="strong-4">Representation Learning</strong></em> in the context of Deep Learning.</p>
<h2 id="h2-2">1. ShapeVAE: Representation Learning for 3D Shape</h2>
<blockquote id="blockquote-1">
<p id="p-17">Goal: Representation Learning for 3D Shape</p>
</blockquote>
<p id="p-18">ShapeVAE is an AutoEncoder (specifically, a Variational AutoEncoder or VAE) defined for 3D shapes.  Like AutoEncoders (VAEs) in other domains used in Latent Diffusion Models (LDMs), the purpose of all ShapeVAEs is the same:</p>
<p id="p-19">"To obtain a semantically meaningful, learnable representation of the input data."</p>
<p id="p-20">Prominent examples include:</p>
<ul id="ul-4">
<li id="li-6"><a id="a-2" href="https://1zb.github.io/3DShape2VecSet/">Shape2vecset</a></li>
<li id="li-7"><a id="a-3" href="https://neuralcarver.github.io/michelangelo/">Michelangelo (Neural Carver)</a></li>
<li id="li-8"><a id="a-4" href="https://craftsman3d.github.io/">Craftsman</a></li>
</ul>
<p id="p-21">All of these share a similar pipeline design.  In essence, it's a standard AutoEncoder structure with the following characteristics:</p>
<figure id="figure-2" >
  <img src='./250302_3d_latent_diffusion/assets/image-24.png' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> ShapeVAE pipeline (from CraftsMan)</figcaption>
</figure>

<ol id="ol-1">
<li id="li-10"><strong id="strong-5">Input</strong>: Point clouds (usually sampled from the ground truth mesh of the training set).</li>
<li id="li-11"><strong id="strong-6">What to Learn</strong>?
a. <strong id="strong-7">Learnable query</strong> (latent vector set).
b. <strong id="strong-8">AutoEncoder</strong> (weights of the linear projection layers in each self/cross-attention block).</li>
<li id="li-12"><strong id="strong-9">Output</strong>: 3D shape, typically represented as occupancy fields (a binary voxel grid).</li>
</ol>
<p id="p-23">Here, the learnable query acts as a kind of basis in a compressed, semantically meaningful space (the latent space) representing the data distribution (3D shapes).</p>

<figure id="figure-3" >
  <img src='./250302_3d_latent_diffusion/assets/image-25.png' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Various <em>Latent</em> Type</figcaption>
</figure>

<p id="p-25">This functions similarly to the kernel basis discussed in the preliminaries.</p>
<p id="p-26">In ShapeVAE, using the VAE and learnable queries, the similarity of the learnable query (the basis in the latent space) is reflected for a query point $(x)$:</p>
<p id="p-27">
$$ \sum_{i=1}^{M} \mathbf{v}(\mathbf{f}_i) \cdot \frac{1}{Z(\mathbf{x}, \{\mathbf{f}_i\}_{i=1}^{M})} e^{\mathbf{q}(\mathbf{x})^{\mathsf{T}} \mathbf{k}(\mathbf{f}_i) / \sqrt{d}}
$$
</p>

<p id="p-28">where Z is a normalizing factor (so, excluding v, it's effectively a softmax). This embedding is then decoded to reconstruct the ground truth shape, thereby learning the latent space.</p>
<blockquote id="blockquote-2">
<p id="p-29">The authors of Shape2vecset, who first introduced ShapeVAE, state that they drew inspiration from RBFs for this design.  Since $q(x)$ is the input embedding, the learnable parameters here are the latent vectors $(f_i)$ and their corresponding weights $(\mathbf{v}(f_i))$. This is analogous to approximating the RBF value using the weighted similarity between the query point and the kernel basis, and then optimizing the basis function using this approximation.</p>
</blockquote>
<p id="p-30">This equation can also be interpreted as a form of <em id="em-7"><strong id="strong-10">QKV cross-attention</strong></em> in a Transformer.  The authors define the learnable latent, drawing inspiration from DETR and Perceiver, as follows:</p>
<p id="p-31">$$ \text{Enc}_{\text{learnable}}(\mathbf{X}) = \text{CrossAttn}(\mathbf{L}, \text{PosEmb}(\mathbf{X})) \in \mathbb{R}^{C \times M}
$$</p>
<p id="p-32">In other words, the ShapeVAE AutoEncoder uses a basis (the latent query, <strong id="strong-11">L</strong>) to encode and decode the relationship between the basis and each data instance (<strong id="strong-12">X</strong>).</p>

<figure id="figure-4" >
  <img src='./250302_3d_latent_diffusion/assets/image-26.png' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> ShapeVAE pipeline (from 3DShape2vecset)</figcaption>
</figure>

<p id="p-34">It learns the <em id="em-8"><strong id="strong-13">latent space</strong></em> that best represents the shape, and the <em id="em-9"><strong id="strong-14">basis (learnable query)</strong></em> that best captures the information about that latent space.</p>
<p id="p-35">The components of the pipeline have the following details:</p>
<ul id="ul-7">
<li id="li-14"><strong id="strong-15">Positional Encoding</strong>: Fourier features, similar to the sinusoidal encoding used in NeRF and Transformers. PE not only maps Cartesian coordinates to a high-dimensional, frequency domain, but also adds stationarity between coordinates when learning kernel regression.<img id="img-7" src="./250302_3d_latent_diffusion/assets/image-12.png" width="70%" /></li>
<li id="li-15"><strong id="strong-16">KL regularization term</strong>: Encourages the latent distribution generated by the encoder to be close to the prior distribution (typically a standard Gaussian distribution, $N(0, 1)$). This provides several advantages:<ul id="ul-8">
<li id="li-16"><strong id="strong-17">Continuous latent space</strong>: A latent space following a normal distribution is continuous and smooth, making interpolation and sampling in the latent space easier.<ul id="ul-9">
<li id="li-17">Shape variations can be naturally controlled through vector operations (interpolation, extrapolation, etc.) in the latent space.</li>
</ul>
</li>
<li id="li-18"><strong id="strong-18">Prevent Overfitting</strong>: By constraining the latent space to be close to the prior distribution, the encoder is encouraged to learn the distribution of the training data in a more general form.</li>
<li id="li-19"><strong id="strong-19">Sampling Ease</strong>: New data can be generated by simply sampling randomly from the standard Gaussian distribution and inputting it to the decoder.</li>
</ul>
</li>
</ul>
<blockquote id="blockquote-3">
<p id="p-36">Q. Why learnable query?
DETR and Perceiver are fundamentally designed for tasks like detection and classification, not generation.  While learnable queries are sometimes used, they are not common in 2D LDMs.  However, the reasons for introducing latent queries in ShapeVAE can be inferred as follows:</p>
<ul id="ul-10">
<li id="li-20"><strong id="strong-20">Hierarchy</strong>: Clear Part-based Structure: 3D shapes often consist of meaningful parts. These parts have spatial relationships and constitute the overall shape.</li>
<li id="li-21"><strong id="strong-21">Sparsity &amp; Geometry</strong>: 3D shapes have sparse characteristics, and the geometry itself is the core information, much sparser than the texture, style, and background of a '2D image'. Therefore, they are well-suited for compression using a latent query approach.</li>
</ul>
</blockquote>
<p id="p-37">While there are differences in the structure used for the encoder/decoder (Perceiver ↔︎ Diffusion Transformer) and whether additional losses are added for multi-modal alignment in the latent space (CraftsMan), the fundamental role of ShapeVAE remains consistent.</p>
<p id="p-38">With a well-trained latent space derived from rich data, we can expect that, leveraging the power of Latent Diffusion Models, a generative model for 3D shapes (a 3D Latent Diffusion Model) can be trained.</p>
<hr id="hr-3" />
<h3 id="h3-1">Challenges for ShapeVAE</h3>
<p id="p-39">However, until recently, 3D generation has not shown the same remarkable results as 2D and video generation. The reasons for this slower progress include:</p>
<ul id="ul-11">
<li id="li-22"><strong id="strong-22">Versatility &amp; Diversity of Data</strong>: The amount of data is extremely limited compared to 2D. Objaverse, one of the larger datasets, contains around 8 million assets, and even its expanded XL version only has around 100 million, significantly less than 2D datasets (LAION-5B has 5 billion...). High-quality datasets are even more scarce.</li>
<li id="li-23"><strong id="strong-23">Curse of Dimensionality</strong>: Being three-dimensional, generating high-resolution outputs is more computationally expensive than in 2D.</li>
<li id="li-24"><strong id="strong-24">What is the ‘BEST’ representation?</strong> The question of the appropriate representation for '3D' is an open problem without a definitive answer. Besides neural fields, there are numerous representations like voxels, occupancy grids, and SDFs, each with its advantages and disadvantages, making it difficult to choose one.</li>
</ul>
<p id="p-40">Furthermore, the popularity of NeRF led to SDS, a combination of 2D Diffusion and NeRF, becoming mainstream in 3D generation.  However, it has been extremely difficult to overcome its inherent limitations (extremely slow generation time and the Janus problem). <em id="em-10">cf</em>: <a id="a-5" href="https://velog.io/@gjghks950/3d">Are NeRF and SDS Destined to be Obsolete in 3D Generation?</a></p>
<p id="p-41">The performance of ShapeVAE is directly related to <em id="em-11">1) the quantity and diversity of data</em>.  Early ShapeVAEs focused more on generative approaches to surface reconstruction than 3D generation, and were therefore trained on limited datasets like ShapeNet.  Even when trained on larger datasets, they struggled to faithfully reproduce input images or text.</p>
<p id="p-42">Another problem with ShapeVAE is that feature aggregation is <em id="em-12">'dependent only on spatial location'</em>.  Basically, point features in ShapeVAE are input as 'Positional Encoding'. However, Positional Encoding alone is insufficient to capture important 3D information such as local structure (curvature, surface normal) and global shape context (connection relationships between vertices and faces).</p>
<p id="p-43"><strong id="strong-25">CLAY-Rodin</strong> actively utilizes ShapeVAE in the geometry generation stage, greatly increasing model and data capacity, and uses a texture synthesis approach for texture generation.</p>

<figure id="figure-5" >
  <img src='./250302_3d_latent_diffusion/assets/image-27.png' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Rodin (CLAY)</figcaption>
</figure>

<p id="p-45">Rodin significantly improves the performance of ShapeVAE and shape generation in its latent space using a large DiT model (1.5B), and achieves results that surpass previous 3D generation quality by using a two-stage approach of generating a shape (mesh) and then generating texture on it using geometry-guided multi-view generation.</p>
<p id="p-46"><img id="img-9" style="width: 100%" src="./250302_3d_latent_diffusion/assets/image-7.png" alt="" /></p>
<ul id="ul-13">
<li id="li-26"><a id="a-7" href="https://ncsoft.github.io/CaPa/">CaPa</a>, with a similar pipeline to Clay. It generates a mesh with a 3D LDM and then backprojects high-quality texture.</li>
</ul>
<p id="p-47">Following this research, the number of studies presenting 3D Latent Diffusion itself gradually increased in 2024.  However, the aforementioned problems:</p>
<ol id="ol-2">
<li id="li-27">Low Quality</li>
<li id="li-28">Not faithfully following the guidance (input image or text)</li>
</ol>
<p id="p-48">remained difficult to solve...</p>
<h2 id="h2-3">2. Trellis</h2>
<p id="p-49">Paper: <a id="a-8" href="https://trellis3d.github.io/">Trellis: Structured 3D Latents for Scalable and Versatile 3D Generation</a></p>
<p id="p-50">Trellis is a state-of-the-art 3D Latent Diffusion model announced by Microsoft in late 2024.  It significantly outperforms previous ShapeVAE-based approaches in terms of instruction following and stability, and has the advantage of generating both shape and texture end-to-end.  Let's analyze the design that enabled it to achieve state-of-the-art quality.</p>
<h3 id="h3-2">2.1. Structured Latent Representation</h3>
<p id="p-52">The authors propose a representation called <em id="em-14"><strong id="strong-27">SLAT</strong></em> (Structured Latent Representation):</p>
<p id="p-53">
$$ \mathbf{z} = \{(\mathbf{z}_i, \mathbf{p}_i)\}_{i=1}^{L}, \quad \mathbf{z}_i \in \mathbb{R}^{C}, \quad \mathbf{p}_i \in \{0, 1, \dots, N-1\}^3,
$$</p>
<p id="p-54">where $p_i$ is the voxel index and $(z_i)$ is the latent vector.  That is, each voxel grid has a latent vector assigned to it, and the latent vector set itself is <em id="em-15"><strong id="strong-28">structured (voxelized)</strong></em>, hence the name <em id="em-16"><strong id="strong-29">SLAT</strong></em>.</p>
<p id="p-55">Due to the sparsity of 3D data, the number of activated grids is much smaller than the total size of the 3D grid $(L &lt;&lt; N^3)$, which means that relatively high-resolution outputs can be generated.  This approach is similar to voxel-grid NeRFs like Instant-NGP, but uses ShapeVAE to predict the featured voxel-grid.</p>
<p id="p-56">While the definition itself might seem like simply mapping the learnable query used in ShapeVAE to a voxel grid, the core of SLAT is that it actively utilizes the <strong id="strong-30">DINOv2</strong> feature extractor during the learning process of this SLAT encoding.</p>
<p id="p-57"><img id="img-10" style="width: 100%" src="./250302_3d_latent_diffusion/assets/image-9.png" alt="" /></p>
<p id="p-58">As shown in the figure above, SLAT calculates the encoding for 3D assets during the VAE training process by:</p>
<ol id="ol-3">
<li id="li-29">Multi-View Rendering.</li>
<li id="li-30">Featurizing: Extracting features for each view rendering using <strong id="strong-71" > DINOv2</strong>.</li>
<li id="li-31">Averaging.</li>
</ol>
<p id="p-51">In other words, Trellis overcomes the limitations of ShapeVAE stemming from the <em id="em-13">'PE-only input feature encoding'</em> by utilizing a well-trained 2D feature extractor (<strong id="strong-26">DINOv2</strong>).</p>
<p id="p-59">This approach seems to be adopted because Trellis aims for <em id="em-17"><strong id="strong-31">end-to-end 3D generation</strong></em>.  It obtains versatile features using <em id="em-18"><strong id="strong-32">pre-trained DINOv2</strong></em> to capture information about 3D assets, such as color and texture, which are difficult to represent with PE alone.</p>


<p id="p-61">The VAE structure itself is the same as the original ShapeVAE. Since the latent space is well-defined, the decoder can be changed to fine-tune the output to generate 3D Gaussian Splattings (GSs), Radiance Fields (NeRFs), or Meshes. Therefore, Trellis can predict outputs in a format-agnostic manner, including GSs, NeRFs, and Meshes. (The actual inference branch uses both GS and Mesh branches).</p>
<h3 id="h3-3">2.2. SLAT Generation</h3>
<blockquote id="blockquote-4">
<p id="p-62">Q.  So, can new assets be generated by simply inputting a random sample from a Standard Gaussian Distribution in the latent space, as in ShapeVAE?</p>
</blockquote>
<p id="p-63">Unfortunately, this is not the case.  First, since SLAT's <em id="em-19"><strong id="strong-33">'structure' (position index)</strong></em> itself is meaningful, it is necessary to generate the structure, i.e., which voxels are empty and which are not.</p>


<figure id="figure-6" >
  <img src='./250302_3d_latent_diffusion/assets/image-28.png' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Trellis's 3D Asset Generation Pipeline</figcaption>
</figure>

<p id="p-65">To achieve this, Trellis uses a two-stage approach in 3D generation.</p>
<ol id="ol-4">
<li id="li-33">Generate Sparse Structure via Conditional Flow Matching: Uses a rectified flow model. <strong id="strong-73" ><em> (model 1)</em></strong>
$$ \mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} || \mathbf{v}_\theta(\mathbf{x}, t) - (\epsilon - \mathbf{x}_0) ||_2^2.
$$</li>
<li id="li-34">Generate the final SLAT using a Transformer similar in structure to DiT, with the generated Sparse Structure as input. <strong id="strong-74" ><em> (model 2)</em></strong></li>
</ol>
<p id="p-66">Directly generating the structure as a dense grid in the first stage is computationally expensive. Therefore, Trellis also adopts a <em id="em-20">Decoder</em> in the Structure Generation Stage, generating a low-resolution feature grid and then scaling it up with the Decoder.</p>
<p id="p-67">The ground truth dense binary grid $\mathbf{O} \in \{0, 1\}^{N \times N \times N}$ is compressed into a low-resolution feature grid $\mathbf{S} \in \mathbb{R}^{D \times D \times D \times C_s}$ using 3D convolution. A Flow Transformer is then trained to predict this compressed representation.</p>
<p id="p-68">Because $\mathbf{O}$ is a coarse shape, there is almost no loss during the compression process using 3D convolution, which also improves the efficiency of neural network training. Furthermore, the binary grid of $\mathbf{O}$ is converted to continuous values, making it suitable for Rectified Flow training.</p>
<blockquote id="blockquote-5">
<p id="p-69">The Rectified flow model employs linear interpolation (input → output) of the straight-line path from data → noise as the forward process in the diffusion process. Compared to inefficient steps in typical Neural ODE solvers, Rectified Flow can model the linear vector field from data → noise, enabling much faster and more accurate generation.</p>
<p id="p-70"><img id="img-13" src="./250302_3d_latent_diffusion/assets/image-3.png" alt="" /></p>
</blockquote>
<p id="p-71">In this process, condition modeling is performed similarly to other Diffusion models, injecting into the KV of cross-attention. In other words, Sparse Structure Generation acts as a kind of <em id="em-21"><strong id="strong-34">Image/Text-to-3D Coarse Shape Generation</strong></em>.</p>
<p id="p-72"><img id="img-14" style="width: 100%" src="./250302_3d_latent_diffusion/assets/image-1.png" alt="" /></p>
<p id="p-73">Subsequently, model 2, which generates SLAT using the generated Sparse Structure, is also trained using Rectified Flow. The decoder of ShapeVAE is then used on the finally generated SLAT to produce GSs / Mesh outputs.</p>
<p id="p-74">The final 3D asset output uses <em id="em-22">RGB representation</em> for GS generation results and <em id="em-23">geometry</em> for Mesh generation results, fitting the mesh texture to the GS rendering. This is likely a strategy employed because it is difficult to convert 3D GS into a clean mesh.</p>
<p id="p-75">Specifically, the steps are:</p>
<ol id="ol-5">
<li id="li-35"><strong id="strong-35">Multi-view Rendering</strong>: Rendering the GSs generation results for a predetermined number of views.</li>
<li id="li-36"><strong id="strong-36">Post-processing</strong>: Post-processing the Mesh generation results using the mincut algorithm for retopology, hole filling, etc.</li>
<li id="li-37"><strong id="strong-37">Texture Baking</strong>: Learning the texture by minimizing the Total-Variation Loss (L1) between the mesh texture and the multi-view GS renderings from step 1) (using them as ground truth textures), and finally baking the texture onto the Mesh.</li>
</ol>
<p id="p-76">The glb outputs available for download on the demo page are all the results of this pipeline.</p>
<ul id="ul-15">
<li id="li-38">cf: <a id="a-9" href="https://github.com/microsoft/TRELLIS/blob/eeacb0bf6a7d25058232d746bef4e5e880b130ff/trellis/utils/postprocessing_utils.py#L399">to_glb</a>, <a id="a-10" href="https://github.com/microsoft/TRELLIS/blob/eeacb0bf6a7d25058232d746bef4e5e880b130ff/trellis/utils/postprocessing_utils.py#L22">fill_holes</a></li>
</ul>
<img id="img-15" style="width: 100%" src="./250302_3d_latent_diffusion/assets/x4-1.png" />

<p id="p-77">It demonstrates outputs of impressive quality.  On the <a id="a-11" href="https://trellis3d.github.io/">project page</a> and <a id="a-12" href="https://huggingface.co/spaces/JeffreyXiang/TRELLIS">Demo</a>, you can see results that are even better than the paper captures.</p>
<p id="p-78">A drawback is that it doesn't yet completely follow instructions (input guidance) perfectly, and perhaps because the default branch of the decoder is GSs, the quality of the generated mesh isn't always excellent.</p>
<hr id="hr-4" />
<h2 id="h2-4">3. Hunyuan3D-v2</h2>
<p id="p-79">Paper: <a id="a-13" href="https://3d-models.hunyuan.tencent.com/">Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation</a></p>
<p id="p-80">After the emergence of Trellis, it seemed that Trellis would firmly hold the throne of SOTA 3D Generation for some time. However, Hunyuan3D-v2 from China has appeared, surpassing Trellis in terms of Mesh Quality and Instruction Following.</p>
<p id="p-81">Unlike Trellis, Hunyuan3Dv2 is not end-to-end but follows a two-stage approach like Rodin or CaPa: 1) Mesh Generation 2) Texture Generation. The <em id="em-24"><strong id="strong-38">Mesh Generation</strong></em> quality is truly <em id="em-25"><strong id="strong-39">SUPERIOR</strong></em>. Let's analyze it in detail.</p>
<h3 id="h3-4">3.1. Hunyuan-ShapeVAE</h3>
<p id="p-82">The design of Hunyuan's ShapeVAE is not very different from vanilla ShapeVAE, but there are some significant differences:</p>
<ol id="ol-6">
<li id="li-39"><p id="p-83"><strong id="strong-40">Point Sampling</strong>: When training ShapeVAE, point clouds are usually obtained from the ground truth mesh through uniform sampling. However, this often leads to the loss of fine details. Therefore, Hunyuan uses a point sampling strategy that focuses more on edges and corners, in addition to uniform sampling. This approach is similar to that of the recently proposed <a id="a-14" href="https://aruichen.github.io/Dora/">Dora</a>.</p>
<p id="p-84"> <img id="img-16" style="width: 100%" src="./250302_3d_latent_diffusion/assets/image-15.png" alt="" /></p>
<p id="p-85"> Figure from Dora. Left: salient points ↔︎ Right: uniform</p>
</li>
<li id="li-40"><p id="p-86"><strong id="strong-41">SDF estimation</strong>: The VAE output is not a binary voxel grid but SDF values. Unlike the existing occupancy method, which requires predicting a binary grid, this allows the deep learning model to estimate continuous real SDF values, resulting in more stable outputs.</p>
</li>
<li id="li-41"><p id="p-87"><strong id="strong-42">Point Query</strong>: It does not use the strategy of learnably learning the basis of the latent space as a latent vector set. Instead, it uses salient/uniform sampled points, subsampled, as queries.</p>
<img id="img-17" src="./250302_3d_latent_diffusion/assets/image-5.png" width="70%" /></li>
</ol>
<p id="p-88">It seems they have adopted a strategy of learning the latent space itself more precisely, rather than learning the basis of the latent space. This approach is likely possible because salient sampling sufficiently reflects the fine details of each 3D shape.</p>
<p id="p-89"><img id="img-18" style="width: 100%" src="./250302_3d_latent_diffusion/assets/image-16.png" alt="" /></p>
<ul id="ul-16">
<li id="li-42">Figure: Hunyuan-ShapeVAE</li>
</ul>
<h3 id="h3-5">3.2. Hunyuan3D-DiT</h3>
<p id="p-90">The core of Hunyuan3D-DiT is the novel architecture design of the 3D generation stage.</p>
<p id="p-91">Previous studies, including Trellis, used Transformers not significantly different from the general DiT structure. However, Hunyuan uses a <em id="em-26"><strong id="strong-43">'double- and single-stream' design</strong></em> like Flux.</p>
<p id="p-92">Although there is no official technical report for Flux, according to the released development version code, it processes information from the text ↔︎ image modalities in a <strong id="strong-44">double-stream</strong> manner. This is considered the main reason why Flux achieves better instruction-following performance compared to SDXL.</p>

<table class="table " id="table-5">
<thead>
<tr>
<th>SDXL</th>
<th>Flux</th>
</tr>
</thead>
<tbody>
<tr>
<td>
    <figure id="figure-7" >
  <img src='./250302_3d_latent_diffusion/assets/image-29.png' alt='img alt' width='100%'>
</figure>
</td>
<td>
    <figure id="figure-8" >
  <img src='./250302_3d_latent_diffusion/assets/image-30.png' alt='img alt' width='100%'>
</figure>
</td>
</tr>
</tbody>
</table>


<ul id="ul-17">
<li id="li-43">cf. <a id="a-16" href="https://www.reddit.com/media?url=https%3A%2F%2Fpreview.redd.it%2Fa-detailled-flux-1-architecture-diagram-v0-ary85pw338od1.png%3Fwidth%3D7710%26format%3Dpng%26auto%3Dwebp%26s%3D9dd2a75cf75bc2dc1d0f1e7b27fb8a5f67253eb1">unofficial diagram of Flux Pipeline</a></li>
<li id="li-44"><strong id="strong-45">My opinion)</strong> The structure of the double stream is similar to the reference net method of ControlNet. I suspect it was inspired by the way ControlNet effectively reflects conditions without compromising the generation capability of the original modal.</li>
</ul>

<figure id="figure-7" >
  <img src='./250302_3d_latent_diffusion/assets/image-31.png' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Illustration of Hunyuan3D-DiT</figcaption>
</figure>

</ul>
<p id="p-95">Hunyuan also adopts this <em id="em-27"><strong id="strong-46">'double-single'</strong></em> structure to generate high-quality 3D shapes while preserving as much information as possible from the condition (image, text) instructions.</p>
<p id="p-96">The core of the pipeline is as follows:</p>
<ul id="ul-19">
<li id="li-46"><strong id="strong-47">Double Stream</strong>:<ul id="ul-20">
<li id="li-47"><strong id="strong-48">Shape Tokens</strong>: Latent representation tokens (noisy) of the 3D shape to be generated are refined through the diffusion reverse process of DiT.</li>
<li id="li-48"><strong id="strong-49">Image Tokens</strong>: 2D image features extracted from the input image (image prompt) using pre-trained DINOv2.</li>
<li id="li-49"><strong id="strong-50">Shared Interaction</strong>: Shape and Image Tokens are processed through separate paths, but interactions between the two tokens are reflected within the Attention operation. This effectively incorporates information from the image prompt into the 3D shape generation process.</li>
</ul>
</li>
<li id="li-50"><strong id="strong-51">Single Stream</strong>:<ul id="ul-21">
<li id="li-51"><strong id="strong-52">Input</strong>: Shape Tokens that have incorporated image information through the double-stream.</li>
<li id="li-52"><strong id="strong-53">Output</strong>: The tokens are processed independently to further refine the 3D shape latent representation and generate the final 3D shape (latent).</li>
</ul>
</li>
</ul>
<p id="p-97">Training, like Trellis, reportedly uses <em id="em-28"><strong id="strong-54">Rectified Flow Matching</strong></em>.</p>
<p id="p-98">
$$ \mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[ || u_\theta(x_t, c, t) - u_t ||_2^2 \right]
$$</p>
<p id="p-99">Among the mentioned training details, it is noteworthy that unlike ViT-based approaches that typically add positional embedding (PE) to each patch, Hunyuan removed PE. This is to prevent specific latents from being assigned to 'fixed locations' during shape generation.</p>
<h3 id="h3-6">3.3. Hunyuan3D-paint</h3>
<p id="p-100">Since Hunyuan uses a 2-stage approach like CLAY/CaPa, it uses <em id="em-29"><strong id="strong-55">Geometry-guided Multi-View Generation</strong></em> for texture synthesis. However, it is not simply a combination of MVDream/ImageDream series models + MV-Depth/Normal ControlNet. Instead, it incorporates several novel strategies to improve quality.</p>
<p id="p-101">First, Hunyuan starts by pointing out the problems with existing methods. MVDream and ImageDream try to achieve <em id="em-30">Multi-View Synchronization in the generation branch using 'noisy features'</em> while tuning the Stable Diffusion model for Multi-View. This can lead to the <em id="em-31"><strong id="strong-56">loss of original details</strong></em> in the reference image. Indeed, looking at the MV output of ImageDream or Unique3D, even the front view often shows degraded quality compared to the input image.</p>
<table id="table-2">
<thead>
<tr>
<th><strong id="strong-57">input</strong></th>
<th><strong id="strong-58">generated front view (MVDiffusion)</strong></th>
</tr>
</thead>
<tbody><tr>
<td><img id="img-20" src="./250302_3d_latent_diffusion/assets/image-10.png" width="300" /></td>
<td><img id="img-21" src="./250302_3d_latent_diffusion/assets/image-11.png" width="300" /></td>
</tr>
</tbody></table>

<figure id="figure-9" >
  <img src='./250302_3d_latent_diffusion/assets/image-32.png' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Illustration of Hunyuan3D-Paint</figcaption>
</figure>


<p id="p-103">To address the shortcomings of existing MVDiffusion, Hunyuan uses the following approaches:</p>
<ul id="ul-23">
<li id="li-54"><p id="p-104"><strong id="strong-59">Clean Input Noise</strong></p>
<ul id="ul-24">
<li id="li-55">The <strong id="strong-60">"original VAE feature"</strong> (clean VAE feature without noise) of the reference image is directly injected into the reference branch to preserve the details of the reference image as much as possible.</li>
<li id="li-56">Since the feature input to the reference branch is noiseless (clean, input of the forward process), the timestep of the reference branch is set to 0.</li>
</ul>
</li>
<li id="li-57"><p id="p-105"><strong id="strong-61">Regularization &amp; Weight Freeze Approach</strong></p>
<ul id="ul-25">
<li id="li-58"><strong id="strong-62">Style Bias Regularization</strong>: To prevent style bias that may occur in datasets rendered from 3D assets, they reportedly abandoned the shared-weighted reference-net structure.</li>
<li id="li-59"><strong id="strong-63">Weight Freeze</strong>: Instead, the <em id="em-32"><strong id="strong-64">weights of the original SD2.1 model are frozen and used as a reference-net</strong></em>. The SD2.1 model serves as the base model for multi-view generation, and the <em id="em-33">'frozen weights act as a regularization'</em>. This is a similar strategy to MVDream, where about 30% of the generation results were trained with a simple text-to-image (not MV) loss (from the LAION dataset) to preserve the fidelity of the MVDiffusion model.</li>
</ul>
<blockquote id="blockquote-6">
<p id="p-106">This might be a bit confusing, but think of it as using the opposite approach of typical ControlNet. The reference branch, which plays the control role, is not trained, and the generation branch (MV-Diffusion model) is trained.  The 'guide' is handled by the original SD model, and 'gen' is handled by the MV-Diffusion model.</p>
</blockquote>
</li>
<li id="li-60"><p id="p-107">For <strong id="strong-65">Geometry Conditioning</strong>, both of the following are used:</p>
<ul id="ul-26">
<li id="li-61"><strong id="strong-66">CNM (Canonical Normal Maps)</strong>: An image of the 3D model surface normal vectors projected onto a canonical coordinate system.</li>
<li id="li-62"><strong id="strong-67">CCM (Canonical Coordinate Maps)</strong>: An image mapping the 3D model surface coordinate vectors to a canonical coordinate system.</li>
</ul>
<blockquote id="blockquote-7">
<p id="p-108">Both project onto a canonical system to provide geometry-invariant information. Using both coordinate and normal information maps both the spatial position and the relationships between positions. MetaTextureGen also uses the same guidance, reporting that the combination of point and normal is better than depth maps in terms of detail and global context.
  <img id="img-23" src="./250302_3d_latent_diffusion/assets/image-13.png" width="70%" /></p>
</blockquote>
</li>
</ul>
<p id="p-109">Finally, <strong id="strong-68">Multi-Task Attention</strong> is proposed to train this structure effectively.</p>
<p id="p-110">Mathematically, this can be expressed as:</p>
<p id="p-111">
$$\begin{aligned}Z_{MVA} = Z_{SA} &+ \lambda_{ref} \cdot \text{Softmax}\left(\frac{Q_{ref} K_{ref}^T}{\sqrt{d}}\right) V_{ref} \\ &+ \lambda_{mv} \cdot \text{Softmax}\left(\frac{Q_{mv} K_{mv}^T}{\sqrt{d}}\right) V_{mv} \end{aligned}
$$</p>
<p id="p-112">As the equation indicates, this is a parallel attention mechanism designed for a form of "multi-task learning," where the reference (Ref) module and the multi-view (mv) module operate independently.</p>
<p id="p-113">This design is motivated by the distinct roles of the reference branch (ControlNet) and the generation branch (MV generation) in the current architecture:</p>
<ul id="ul-27">
<li id="li-63"><strong id="strong-69">Reference branch</strong>: Aims to adhere to the original image.</li>
<li id="li-64"><strong id="strong-70">Generation branch</strong>: Aims to maintain consistency between generated views.</li>
</ul>
<p id="p-114">The Multi-Task Attention mechanism is intended to mitigate the conflicts and resulting performance degradation that can arise from this multi-functionality.</p>
<p id="p-115">A similar structural design has been demonstrated in <a id="a-17" href="https://huanngzh.github.io/MV-Adapter-Page/">MV-Adapter</a>. Both cases employ this design to achieve multi-view generation capabilities without sacrificing the performance of the original branch. In a sense, this approach is analogous to the double-stream architecture of the ShapeGen stage.</p>
<p id="p-116"><img id="img-24" style="width: 100%" src="./250302_3d_latent_diffusion/assets/image-14.png" alt="" /></p>
<p id="p-117">This architecture enables a diffusion model design that leverages the reference image as guidance while simultaneously ensuring multi-view consistency. In other words, it allows for the generation of natural images from various viewpoints while maintaining consistency with the reference image.</p>
<p id="p-118">Tests have shown that this approach produces MVDiffusion outputs with quality significantly surpassing the fidelity of existing MVDiffusion methods. Furthermore, it exhibits higher multi-view consistency and fewer seams and artifacts compared to competitors that rely solely on normal or depth maps for guidance.</p>
<img id="img-25" style="width: 100%" src="./250302_3d_latent_diffusion/assets/image-6.png" alt="" />
<ul id="ul-28">
<li id="li-65">Figure from MV-Adapter. An ablation study demonstrating the effectiveness of the parallel structure.</li>
</ul>
<p id="p-120">In my own testing, truly superior-quality meshes were generated, and the geometry-guided MV Diffusion results also exhibited higher fidelity and consistency than any previous model I've encountered.</p>

<p id="p-115" >Test results showed MVDiffusion output quality that overwhelmingly surpasses the fidelity of existing MVDiffusion, and it exhibited higher multi-view consistency and fewer seams and artifacts than competitors that simply use normal/depth maps as guides.</p>


<table id="table-6"  class="table ">
<thead>
<tr>
<th>Hunyuan</th>
<th>ImageDream</th>
</tr>
</thead>
<tbody>
<tr>
<td><img src='./250302_3d_latent_diffusion/assets/image-17.png' width=100%></td>
<td><img src='./250302_3d_latent_diffusion/assets/image-18.png' width=100%></td>
</tr>
</tbody>
</table>
<br/>

<h2 id="h2-6" > 4. Trellis vs Hunyuan? </h2>
<table id="table-13"  class="table ">
<thead>
<tr>
<th>Trellis</th>
<th>Hunyuan</th>
</tr>
</thead>
<tbody>
<tr>
<td><figure id="figure-10" >
  <img src='./250302_3d_latent_diffusion/assets/image-19.gif' alt='img alt' width='100%'>
</figure></td>
<td>
    <figure id="figure-11" >
  <img src='./250302_3d_latent_diffusion/assets/image-20.gif' alt='img alt' width='100%'>
</figure>
</td>
</tr>
<tr>
<td>
    <figure id="figure-12" >
  <img src='./250302_3d_latent_diffusion/assets/image-21.gif' alt='img alt' width='100%'>
</figure>
</td>
<td>
    <figure id="figure-13" >
  <img src='./250302_3d_latent_diffusion/assets/image-22.gif' alt='img alt' width='100%'>
</figure>
</td>
</tr>
</tbody>
</table>


<p id="p-116" >During my own tests, in terms of Mesh Quality, Hunyuan3D&#39;s topology is significantly better than Trellis. Furthermore, although not included in the blog post, Trellis sometimes predicts results that differ from the input image guidance, while Hunyuan consistently demonstrates faithful instruction following.</p>
<p id="p-117" >On the other hand, Texture Quality is not yet very high for either model. Hunyuan generates textures by generating 6 view Multi Views geometry-guided and backprojecting them, which tends to result in some occlusion. Trellis exhibits less occlusion than Hunyuan, but its fidelity is comparatively worse. Additionally, in the case of Hunyuan, it does not perfectly align with the geometry guide, sometimes making seams or artifacts more noticeable than in Trellis.</p>
<p id="p-118" >The clear advantages and disadvantages seem to arise from their respective end-to-end vs. 2-stage pipelines. It is anticipated that subsequent research in 3D Latent Diffusion will emerge, enhancing quality in each of these aspects.</p>
<p id="p-119" >Finally, I show CaPa's Result :)</p>
<figure id="figure-14" >
  <img src='./250302_3d_latent_diffusion/assets/image-23.gif' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Generated 3D asset by CaPa</figcaption>
</figure>


<br/>

<h2 id="h2-5">Closing</h2>
<p id="p-122">Thus far, I have provided a detailed analysis tracing the evolution of state-of-the-art 3D Latent Diffusion, from the fundamental concepts of ShapeVAE to Trellis and Hunyuan3D.</p>
<p id="p-123">While the open-source community did not achieve remarkable progress in the 3D field for some time after the emergence of CLAY, recent studies have showcased innovative designs and reached state-of-the-art quality, further fueling anticipation for generative models in the 3D domain.</p>
<p id="p-124">Personally, Hunyuan's application of proven designs from Flux, MV-Adapter, and other works to the 3D generation scheme is particularly impressive. It reinforces the notion that to conduct impactful research, one must remain attentive to research trends in other fields.</p>
<p id="p-125">Finally, recent research, led by MeshAnything, is attracting attention by focusing on the auto-regressive generation of mesh faces to create what are termed "Artistic-Created Meshes" (these studies also utilize the ShapeVAE latent space). However, due to its auto-regressive nature, this approach is time-consuming, and the quality is not yet satisfactory; therefore, it seems prudent to observe its development for the time being.</p>


<hr/>
<p>
    You may also like, 
</p>
<ul>
    <li>
        <a href="/blogs/posts/?id=240426_diffusion_depth">
            <span style="text-decoration: underline;">Is Diffusion's Estimated Depth Really Good?</span>
        </a>
    </li>
    <li>
        <a href="/blogs/posts/?id=240805_gs">
            <span style="text-decoration: underline;">A Comprehensive Analysis of Gaussian Splatting Rasterization</span>
        </a>
    </li>
</ul>
<br/>