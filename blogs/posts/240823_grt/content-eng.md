title: Don't Rasterize, But Ray Trace 3D Gaussian
date: August 23, 2024
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
    <ul>
        <li>
            <a href="#introduction"> Introduction</a>
        </li>
        <li><a href="#preliminary"> Background</a></li>
        <ul>
            <li>
                <a href="#parameterization"> 3D Gaussian Modeling </a>
            </li>
            <li>
                <a href="#hardware-accelerated-ray-tracing"> About Ray Tracing </a>
            </li>
        </ul>
        <li>
            <a href="#method"> 3D Gaussian Ray Tracing </a>
        </li>
        <ul>
            <li>
                <a href="#bounding-primitives"> Bounding Primitives </a>
            </li>
            <li>
                <a href="#ray-tracing-renderer"> Ray Tracing Renderer </a>
            </li>
            <li>
                <a href="#ray-gaussian-intersection"> Ray Gaussian Intersection </a>
            </li>
        </ul>
        <li><a href="#experiments"> Experiments </a></li>
        <ul>
            <li>
                <a href="#quantitative-results"> Quantitative Results </a>
            </li>
            <li>
                <a href="#qualitative-results"> Qualitative Results </a>
            </li>
        </ul>
        <li><a href="#conclusion"> Closing </a></li>
    </ul>
</nav>

<br/>
<h2 id="tl-dr">TL; DR</h2>
<p class="lang eng">This article provides an in-depth review of the paper &quot;3D Gaussian Ray Tracing,&quot; which introduces a novel approach to leveraging ray tracing in 3D Gaussian Radiance Fields.</p>
<p class="lang eng">3D Gaussian Splatting is a powerful and fascinating technology, but it inherits several problems from rasterization. The recently presented 3D Gaussian Ray Tracing (3D GRT) resolves many of these shortcomings by introducing a Differentiable Ray Tracer for 3D Gaussians.</p>
<p class="lang eng"> Let’s deep dive into the 3D GRT!</p>
<ul>
    <li> project page: <span style="text-decoration: underline;"><a href="https://gaussiantracer.github.io/"> Link </a></span> </li>
</ul>
<figure>
    <img src="./240823_grt/assets/teaser.gif" alt="Gaussian RT" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong></strong> 3D Gaussian Ray Tracing</figcaption>
</figure>
<br/>

<h2 id="introduction"> 1. Introduction</h2><br/>
<h3 id="challenges-in-3d-gaussian-splatting">Challenges in 3D Gaussian Splatting</h3>
<p class="lang eng">
    3D Gaussian Splatting (3D GS) has emerged as a promising approach for high-fidelity novel-view synthesis and real-time rendering, leveraging sophisticated tile-based rasterization. Despite its potential, this area—often referred to as the next frontier in photogrammetry—continues to face several significant challenges.
</p>
<p class="lang eng">A major limitation of 3D GS stems from its reliance on rasterization, which introduces several constraints:</p>

<ol>
    <li>
        <p class="lang eng">
            <strong>Inflexibility with Diverse Camera Models</strong><br/>
            One of the primary limitations of 3D Gaussian Splatting is its inflexibility in accommodating various camera models.
            As highlighted in <span style="text-decoration: underline;"><a href="https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362">previous article</a></span>, the use of EWA splatting introduces affine projection errors. 
            These errors complicate achieving high-quality results, even when modeling non-pinhole camera types.
        </p>
        <figure>
            <img class="img-fluid" src="./240823_grt/assets/cam_model.png" width="85%">
            <figcaption style="text-align: center; font-size: 15px;"> Inflexibility of 3D GS with Diverse Camera Models, source: <span style="text-decoration: underline;"><a href="https://letianhuang.github.io/op43dgs/">Optimal GS</a></span> </figcaption>
        </figure>
    </li>
    <li>
        <p class="lang eng">
            <strong> Sensitivity to Image Quality</strong><br/> 
            Unlike NeRF, which utilizes MLPs and exhibits a degree of robustness against calibration discrepancies between images, 3D GS relies on explicit geometric primitives. This reliance renders 3D GS highly sensitive to variations in image quality, including issues such as motion blur and rolling shutter effects, which can significantly degrade the final output.
        </p>
        <figure>
            <img class="img-fluid" src="./240823_grt/assets/cap.png" width="85%">
            <figcaption style="text-align: center; font-size: 15px;"> <strong>Left</strong>: GT,  <strong>Right</strong>: trained 3D GS on motion blurred scene, <br/> source: <span style="text-decoration: underline;"><a href="https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/">Deblurring 3D GS</a></span> </figcaption>
        </figure>
    </li>
    <li>
        <p class="lang eng">
            <strong>Lack of Physically-Based Rendering Capabilities</strong><br/>
            Another challenge facing 3D GS is its inability to incorporate physically-based rendering (PBR) effects. Since 3D GS does not adhere to the principles of PBR, accurately modeling lighting and reflection effects within a scene remains problematic. This limitation restricts the realism and applicability of 3D GS in scenarios where accurate light interaction is critical.
        </p>
        <figure>
            <img class="img-fluid" src="./240823_grt/assets/ref.png" width="70%">
            <figcaption style="text-align: center; font-size: 15px;"> Left: GT, Right: 3D GS, source: <span style="text-decoration: underline;"><a href="https://gapszju.github.io/3DGS-DR/">3D GS-DR</a></span> </figcaption>
        </figure>
    </li>
</ol>
<p class="lang eng">To address some of these challenges, RadSplat proposes a two-stage learning process. In this approach, Radiance Fields are first learned using Zip-NeRF, which generates perfectly calibrated pinhole images within the NeRF scene. These images are then used as training data for 3D GS.</p>
<p class="lang eng">However, this method is inefficient due to its two-stage nature and fails to resolve the fundamental limitations posed by rasterization, particularly with physically-based rendering.</p>

<br/>

<h2 id="preliminary"> 2. Background </h2>
<br/>
<h3 id="parameterization">2.1. Parameterization</h3>
<p class="lang eng">The primitive kernel in this method is defined using the covariance matrix in 3D space, consistent with the original 3D GS approach. </p>
<div class="math-container">
    $$G(x) = \exp \left( {- \frac{1}{2} x^{\rm T} \Sigma^{-1} x} \right )$$
</div>

<p class="lang eng">Due to this shared kernel definition, most calculations remain similar between the two methods. However, there is a notable difference concerning the direction used when calculating Spherical Harmonics to RGB (SH2RGB).</p>


<p class="lang eng">In 3D GS, the direction is derived from the camera position $o$ and the Gaussian means $\mu$, which is then utilized in SH2RGB calculations.</p> 
<div class="math-container">
    $$ \frac{\mu - \mathbf{o}}{\| \mu - \mathbf{o} \|} $$
</div>
<p class="lang eng">This approach, however, results in a direction that slightly deviates from the actual angle projected onto the pixel. </p> 

<pre><code class="language-cpp" style="font-size: 16px;">// in preprocessCUDA
glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
rgb[idx * C + 0] = result.x;
rgb[idx * C + 1] = result.y;
rgb[idx * C + 2] = result.z;</code></pre>
<ul class="lang eng">
    <li>The precomputed RGB is then input to the Render kernel.</li>
</ul>

<p class="lang eng">The reason for not using the precise ray direction is that color values are pre-computed and stored for use during tile-wise rasterization. This pre-computed RGB data is subsequently fed into the render kernel, optimizing rendering speed.</p>
<p class="lang eng">While this method enhances rendering performance, it compromises the ability to accurately model illumination effects—one of the inherent weaknesses of 3D GS. To address this issue, the 3D GRT utilizes the actual ray direction in SH2RGB calculations, improving illumination effect modeling.</p>

<h3 id="hardware-accelerated-ray-tracing"> 2.2. Hardware-Accelerated Ray Tracing</h3>
<div class="lang eng">
    <p>NVIDIA GPUs, particularly those in the RTX series, are equipped with dedicated RT cores designed for ray tracing. These RT cores handle the intersection calculations between rays and particles, while the more computationally demanding tasks, such as shading, are assigned to the Streaming Multiprocessors (SMs), optimizing overall performance.</p>
    <p>However, existing ray tracers are typically optimized for rendering opaque particles. This means that during ray traversal, the expected hit count is low, and interaction between the SMs and RT cores is minimized. </p>
    <p>Since 3D Gaussian Splatting involves semi-transparent particles, conventional ray tracers are inefficient in this context. The semi-transparency of 3D Gaussian Splatting increases the complexity of ray tracing, requiring more sophisticated handling of ray-particle intersections to achieve efficient and accurate rendering.</p>
</div>

<h2 id="method"> 3. 3D Gaussian Ray Tracing </h2>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig1.png" width="100%" style="cursor: pointer;" data-bs-toggle="modal" data-bs-target="#fig1_modal">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 1.</strong> Method Overview</figcaption>
</figure>
<!-- Modal Structure -->
<div class="modal fade" id="fig1_modal" tabindex="-1" aria-labelledby="imageModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-xl modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-body">
                <!-- Original size image -->
                <img src="./240823_grt/assets/fig1.png" alt="Full Size Image" width="100%" class="img-fluid">
            </div>
        </div>
    </div>
</div>
<p class="lang eng">To effectively design a ray tracer tailored for 3D Gaussian Splatting, two key elements are essential:</p>
<ol class="lang eng">
    <li><p><strong>BVH with Appropriate Proxy Primitives</strong><br/>Use Bounding Volume Hierarchy (BVH) to accelerate hit traversal by defining proxy primitives that encapsulate 3D Gaussians accurately.</p>
    </li>
    <li><p><strong>Rendering Algorithm</strong><br/> Develop a rendering algorithm that casts rays and gathers information specific to 3D Gaussian Ray Tracing, optimizing the process for the unique characteristics of Gaussian splats.</p>
    </li>
</ol>

<br/>

<h3 id="bounding-primitives"> 3.1. Bounding Primitives</h3>
<p class="lang eng">Let&#39;s start with BVH (<span style="text-decoration: underline;"><a href="https://en.wikipedia.org/wiki/Bounding_volume_hierarchy">Bounding Volume Hierarchy</a></span>).</p>

<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig3.png" width="90%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 2.</strong> Bounding Volume Hierarchy</figcaption>
</figure>
<p class="lang eng">BVH is a hierarchical tree structure used to efficiently divide space for rendering and ray tracing. In this structure, parent nodes consist of larger bounding volumes that encompass smaller leaf nodes, facilitating efficient space partitioning and exploration.</p> 
<p class="lang eng">The main objective of BVH in this context is to define a proxy primitive that accurately encapsulates 3D Gaussians and to use this proxy geometry to construct a BVH. This hierarchy then guides the ray traversal process by determining which 3D Gaussians should be considered for intersection tests.</p>

<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig2.png" width="65%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 3.</strong> bounding primitive </figcaption>
</figure>
<p class="lang eng">NVIDIA OptiX, a common framework for ray tracing, offers three predefined proxy primitive types: triangles, spheres, and axis-aligned bounding boxes (AABBs). However, none of these are ideal for 3D Gaussians. For instance, using AABBs would simplify calculations but would lead to many false-positive proxy hits, as AABBs cannot tightly enclose the Gaussian distribution, leading to inefficiencies in ray tracing (see Fig. 4)</p>


<h4 id="stretched-polyhedron-proxy"> Stretched Polyhedron Proxy</h4>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig4.png" width="90%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 4.</strong> Proxy Primitives</figcaption>
</figure>
<p class="lang eng">After experimental evaluations, the authors found that using an icosahedron—a polyhedron with 20 triangular faces—was the most appropriate proxy geometry for 3D Gaussians.</p>
<p class="lang eng">The benefits of using an icosahedron include:</p>
<ul class="lang eng">
    <li><strong>Efficient Ray-Face Intersection Calculation</strong>: Since the icosahedron consists of triangular faces, the intersection tests between rays and these faces are optimized at the hardware level.</li>
    <br/>
    <li><strong>Accurate Wrapping</strong>: The icosahedron can wrap around a 3D Gaussian distribution effectively, minimizing both false positives and false negatives.</li>
</ul>
<p class="lang eng">For an icosahedron inscribed in a unit sphere, the proxy geometry is computed by transforming each vertex using the following formula:</p>



<p>
$$
v \leftarrow v \sqrt{2 \log (\sigma / \alpha_{\min})} \ {\rm SR^T} + \mu 
$$</p>
<p class="lang eng">To break down this formula:</p>
<ol class="lang eng">
    <li><p><strong>Stretching</strong></p>
        <p>The transformation matrix ${\rm SR^{T}}$ and mean vector $\mu$ adjust the icosahedron to fit the local coordinates of the 3D Gaussian. This involves stretching, rotating, and translating the initial icosahedron to properly enclose the 3D Gaussian distribution.</p>
    </li>
    <li><p><strong>Adaptive Clamping</strong></p> 
        <p> The scaling term $\sqrt{2 \log (\sigma / \alpha_{\min})}$ determines how the icosahedron is scaled. Specifically, the parameter $\alpha_{\min}$ (set to 0.01) represents the minimum response value.</p> 
        <p> Though the term is a little bit tricky when considering identical scaling, this relationship simplifies to:</p>
        <div class="math-container">
            $$ \sigma / \alpha_{\min} = \exp(0.5) \ \Rightarrow \sigma \cdot \exp (- 1/2) = \alpha_{\min} $$
        </div>
        <p> Doesn&#39;t the right side look familiar? This expression closely resembles the response function of the Gaussian Splatting. </p>
        <div>
            $$  f_i(p) = \sigma_i \cdot \exp \left( - \frac{1}{2} (\mu_i -p)^{\rm T} \Sigma_i^{-1} (\mu_i - p ) \right ) $$
        </div>
        <p> The scaling factor adjusts the icosahedron to match the point where the response in the Gaussian distribution drops to $\alpha_{\min}$, effectively clamping the scale at a confidence interval where the standard deviation equals 1.</p>
        <p> In my opinion, the choice of $\alpha_{\min} = 0.01$ is justified by calculating the Gaussian pdf at a standard deviation of 2.6, which corresponds to approximately 99% confidence, yielding a value close to 0.01. </p>
        <div class="math-container">
            $$ \exp\left(-\frac{1}{2}(2.6)^2 \right) \cdot \frac{1}{\sqrt{2 \pi}} \approx 0.01 $$
        </div>
        </p>
        <p> Similarly, 3D GS uses a scaling factor equivalent to three times the standard deviation to compute the radius for culling.</p>
        <p> Adaptive Clamping allows for the scaling of the proxy primitive to be small for nearly transparent particles and larger for more opaque ones, improving the accuracy and efficiency of the ray tracing process.</p>
    </li>
</ol>


<p class="lang eng">This approach enables a more efficient and accurate ray-tracing mechanism for 3D Gaussians by using an optimized proxy geometry that is both computationally feasible and tightly conforms to the Gaussian distribution.</p>


<h3 id="ray-tracing-renderer"> 3.2. Ray Tracing Renderer</h3>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig5.png" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 5.</strong> Ray Tracing </figcaption>
</figure>
<p class="lang eng">For differentiable and efficient rendering in 3D GRT, the process involves sequentially rendering through a <em>next $k$ closest hit</em>. This approach helps in managing multiple semi-transparent particles along a single ray path. The rendering process is outlined as follows:</p>
<ol class="lang eng">
    <li><p><strong>Track Particles Using BVH</strong><br/>The next $k$ closest particles along a ray path are tracked using the BVH. At this stage, the hit response (i.e., the particle&#39;s contribution to the final image) is not yet measured.</p>
    </li>
    <li><p><strong>Measure Hit Response Iteratively</strong><br/> Once the $k$ particles are identified, the actual hit response for each particle is measured iteratively within each chunk of the $k$-buffer. This step involves checking all particles that intersect with the ray.</p>
    </li>
    <li><p><strong>Proxy Hit Verification</strong><br/> During the response measurement, all proxy-hit particles along the ray are checked to determine their actual contribution based on their proximity and alignment with the ray.</p>
    </li>
    <li><p><strong>Rendering Termination</strong><br/> The rendering process continues until a certain threshold is reached, beyond which additional particle contributions are negligible, and rendering can be stopped.</p>
    </li>
</ol>
<p class="lang eng">The following diagram (referenced in the text) illustrates the ray tracing process of 3D GRT when $k=3$, showing how multiple particles are managed and rendered along a single ray.</p>

<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig6.png" width="80%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 6. Next $k$ closest hit Ray Tracer:</strong> on each round of tracing, the next $k$ closest hit particles are collected and sorted into depth order along the ray, the radiance is computed in-order, and the ray is cast again to process the next chunk.  </figcaption>
</figure>
<br/>

<h3 id="ray-gaussian-intersection"> 3.3. Ray-Gaussian Intersection</h3>
<p class="lang eng">To calculate the contribution of each particle during ray tracing, 3D GRT determines the point where the particle&#39;s response (or contribution to the final rendered image) is maximized. This is achieved through the following mathematical formulation:</p>

<div class="math-container">
    $$ \tau_{\max} = \frac{(\mu - \mathbf{o})^{\rm T} \Sigma^{-1} \mathbf{d}}{\mathbf{d}^{\rm T} \Sigma^{-1} \mathbf{d} } = \frac{-\mathbf{o}_g^{\rm T} \mathbf{d}_g}{\mathbf{d}_g^{\rm T}\mathbf{d}_g}
    $$
</div>
<p> where $ \mathbf{o}_g = {\rm S^{-1}R^T}(\mathbf{o} - \mu),  d_g = {\rm S^{-1}R^T} \mathbf{d}$. </p>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/inter.png" width="40%">
</figure>

<div class="lang eng">
    <p>Let&#39;s interpret this step by step!</p>
    <ol>
        <li><p><strong>Transformation to Local Coordinates</strong></p>
            <p> The variables $\mathbf{o}_g$ and $\mathbf{d}_g$ represent the ray origin and direction, which is transformed into the local coordinate system of the 3D Gaussian (just as the proxy primitive is defined in these local coordinates).</p>
        </li>
        <li><p><strong>Maximizing Gaussian Density</strong></p>
            <p> The density of the 3D Gaussian can be expressed as a 1D Gaussian along the ray:</p>
            <div class="math-container">
                $$ G(x_g) = \exp\left(-\frac{1}{2} \mathbf{x}_g^{\rm T} \mathbf{x}_g\right) \quad \text{where } \mathbf{x}_g = \mathbf{o}_g + t\mathbf{d}_g
                $$
            </div>
            <p> Since $\exp(-x)$ is a decreasing function, the maximum density corresponds to the minimum value of the inner quadratic term $\mathbf{x}_g^{\rm T} \mathbf{x}_g$.</p>
        </li>
        <li><p><strong>Optimization</strong> </p>
            <p> The problem of finding the maximum density is equivalent to solving the following optimization problem:</p>
            <div class="math-container">
                $$ \min_t \ (\mathbf{o}_g + t \mathbf{d}_g)^T (\mathbf{o}_g + t \mathbf{d}_g) . 
                $$
            </div>
            <p> Since this is a convex function with respect to $t$, the maximum can be found by setting the derivative with respect to $t$ to zero:</p>
            <div class="math-container">
                $$ \begin{aligned} \nabla_t f(t) &= \frac{d}{dt} \left( (\mathbf{o}_g + t \mathbf{d}_g)^T (\mathbf{o}_g + t \mathbf{d}_g)\right) \\ &= 2 \mathbf{d}_g^T (\mathbf{o}_g + t \mathbf{d}_g). \end{aligned}
                $$
            </div>
            <p> Subsequently, the analytic solution can be derived as follows: </p>
            <div class="math-container"> 
                $$ 2 \mathbf{d}_g^T (\mathbf{o}_g + t \mathbf{d}_g) = 0 \\ \rightarrow t = -\frac{\mathbf{o}_g^{\rm T} \mathbf{d}_g}{\mathbf{d}_g^{\rm T}\mathbf{d}_g}
                $$ 
            </div>
            <p> This equation represents the point along the ray where the Gaussian density, and therefore the particle&#39;s contribution to the final image, is maximized.</p>
        </li>
    </ol>
    <p>Intuitively, The closer the ray direction $\mathbf{d}_g$ is to the origin of the 3D Gaussian, the higher the response or contribution from that particle will be. </p>
    <p>Note that, even though ray tracing is performed in the order of proxy hits, the approximation using this method does not significantly degrade performance, despite any slight differences between proxy hit order and actual maximum response order. </p>
</div>

<br/>

<h2 id="experiments"> 4. Experiments</h2><br/>
<h3 id="quantitative-results"> 4.1. Quantitative Results</h3>
<p class="lang eng">The quantitative evaluations of 3D GRT indicate that there is almost no significant difference between the quantitative metrics of 3D GRT and other novel view synthesis (NVS) techniques. While the fps is slightly lower in comparison, 3D GS still achieves real-time performance.</p>

<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig7.png" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 7.</strong> Quantitative Results </figcaption>
</figure>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig8.png" width="70%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 8.</strong> Speed Comparison </figcaption>
</figure>
<br/>

<h4 id="ablation-study-and-ray-tracer-design"> Ablation Study</h4>
<p class="lang eng">
    The paper also explores the design of the Next $k$-closest Ray Tracer (Fig.8 top left), 
    validation for the proxy mesh design (Fig. 8 bottom left) and the determination 
    of an optimal $k$ value in the $k$-buffer (Fig. 8 top right). 
</p>
<p class="lang eng"> Experimental results support the design of the Ray Tracer, highlighting the importance of these parameters in achieving efficient and accurate rendering.</p>

<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig9.png" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 9.</strong> Ablation Study </figcaption>
</figure>
<br/>
<h4> Particle Kernel Design </h4>

<p class="lang eng">
    Since the particle kernel for the designed Ray Tracer does not need to be strictly a 3D Gaussian, the authors experimented with different kernel designs. 
    The four kernels evaluated include:
</p>

<ul>
    <li>
        <p>
            <strong>3D Gaussian</strong>
            <p>
                $$ \hat{p}(x) = \sigma e^{ -(x-\mu)^{\rm T} \Sigma^{-1} (x-\mu) }
                $$ 
            </p>
        </p>
    </li>
    <li>
        <p>
            <strong>Generalized Gaussian</strong>: generalized of degree $n$
            <p>
                $$ \hat{p}_n(x) = \sigma e^{- \left((x-\mu)^{\rm T} \Sigma^{-1} (x-\mu)\right)^n }
                $$ 
            </p>
        </p>
    </li>
    <li>
        <p>
            <strong>2D Gaussian</strong>: Gaussian Surfels, suggested in 2D Gaussian Splatting (cf. <a href="../240602_2dgs/"><span style="text-decoration: underline;">my previous review </span></a>)
        </p>
    </li>
    <li>
        <p>
            <strong>Cosine wave modulation</strong>: aims to model a particle with spatially varying radiance
            <p>
                $$ \hat{p}_c(x) = \hat{p}(x) \left (  0.5 + 0.5 \cos (\psi {\rm R^T S^{-1}} (x-\mu))  \right )
                $$
            </p>
        </p>
    </li>
</ul>
<p class="lang eng">
    As shown in Figure 10, the reconstruction performance is similar across all kernels tested. However, when using the Generalized Gaussian (GG) kernel, 
    the frames per second (fps) nearly double compared to the standard 3D Gaussian kernel.
</p>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/kernel_qual.png" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 10.</strong> Kernel Design Comparison </figcaption>
</figure>
<p class="lang eng">
    This increase in fps is due to the GG kernel’s design, which makes the density more concentrated around the mean. As the density is modeled closer to an opaque particle, the number of ray-particle intersections decreases, thereby improving rendering efficiency. 
</p>
<p class="lang eng">
    This effect is also evident in the ray-hit visualization provided by the authors, which shows fewer ray-particle interactions for the GG kernel.
</p>

<figure>
    <img class="img-fluid" src="./240823_grt/assets/ray_hit.png" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 11.</strong> Ray hit count for left: 3D G, right: GG  </figcaption>
</figure>

<br/>

<h3 id="qualitative-results"> 4.2. Qualitative Results</h3>
<p class="lang eng">In addition to the quantitative analysis, the qualitative results demonstrate how this method effectively overcomes the limitations of rasterization.</p> 
<p class="lang eng"> Specifically, the 3D GRT shows significant improvements in modeling and rendering, particularly in handling complex light effects across various camera models. This ability to accurately represent lighting and reflections, which are often challenging in rasterization-based techniques, demonstrates that the method could be highly effective in realistic rendering scenarios.</p>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig10.png" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 12.</strong> 3D GRT w/ various light effect </figcaption>
</figure>
<figure>
    <img class="img-fluid" src="./240823_grt/assets/fig11.png" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 13.</strong> 3D GRT's reconstruction capability for non-pinhole camera </figcaption>
</figure>
<p class="lang eng">Overall, the combination of both quantitative and qualitative evaluations highlights the strengths of this new 3D GS approach, especially in terms of performance, memory efficiency, and the ability to handle complex visual effects.</p>
<br/>

<h2 id="conclusion"> 5. Conclusion</h2>
<div class="lang eng">
    <p>This paper presents a comprehensive exploration of the differences and advantages of using a ray tracing-based renderer for 3D Gaussian Splatting compared to traditional rasterization techniques. </p>
    <p>While rasterization excels in speed, especially for primary rays from pinhole cameras, 3D GRT offers greater flexibility and generality. It enables advanced rendering effects such as reflections, refractions, depth of field, and complex camera models, which are difficult or impossible to achieve with rasterization.</p>
    <p>The ray tracing approach significantly broadens the scope of 3D GS, allowing for more accurate modeling of general lighting, image formation, and sub-pixel behaviors. It also facilitates the exploration of global illumination, inverse lighting, and physically-based surface reflection models, paving the way for new research directions in these areas.</p>
    <p>However, the inherent trade-offs between the two methods are evident. While rasterization remains faster in scenarios involving primary rays and static scenes, 3D Gaussian Ray Tracing, despite being carefully optimized for hardware acceleration, still requires more computational resources, particularly when frequent BVH rebuilds are necessary for dynamic scenes.</p>
</div>

<hr/>
<p>
    You may also like, 
</p>
<ul>
    <li>
        <a href="/blogs/posts/?id=240805_gs">
            <span style="text-decoration: underline;">A Comprehensive Analysis of Gaussian Splatting Rasterization</span>
        </a>
    </li>
    <li>
        <a href="/blogs/posts/?id=240602_2dgs">
            <span style="text-decoration: underline;">Under the 3D: Geometrically Accurate 2D Gaussian Splatting </span>
        </a>
    </li>
</ul>
<br/>