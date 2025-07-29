title: Under the 3D: Geometrically Accurate 2D Gaussian Splatting
date: June 02, 2024
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
        <li><a href="#intro"> Introduction </a></li>
        <li><a href="#1-preliminary"> Background </a></li>
        <ul>
            <li><a href="#1-1-3d-gaussian-splatting"> 3D Gaussian Splatting </a></li>
            <li><a href="#1-2-surface-reconstruction-problem-in-3d-gs"> Surface Reconstruction Problem </a></li>
            <li><a href="#1-3-sugar-surface-aligned-gaussian-splatting"> Surface-Aligned Gaussian Splatting </a></li>
        </ul>
        <li><a href="#2-2d-gaussian-splatting"> 2D Gaussian Splatting </a></li>
        <ul>
            <li><a href="#2-1-2d-gaussian-modeling-gaussian-surfels-"> 2D Gaussian Modeling </a></li>
            <li><a href="#2-2-splatting"> 2D-to-2D Projection </a></li>
            <li><a href="#2-3-training-2d-gs"> Training 2D GS </a></li>
        </ul>
        <li><a href="#3-experimens-custom-viser-viewer"> Experimens &amp; Custom Viser Viewer </a></li>
        <ul>
            <li><a href="#3-1-qualitative-results-custom-object-reconstruction"> Qualitative Results </a></li>
            <li><a href="#3-2-custom-viser-viewer-for-2d-gaussian-splatting"> Custom Viser Viewer for 2D GS </a></li>
        </ul>
        <li><a href="#4-conclusion"> Conclusion </a></li>
    </ul>
</nav>


<br/>
<h2 id="intro"> Introduction </h2>
<blockquote class="lang eng">
    <p>
        This article provides an in-depth review of the paper "2D Gaussian Splatting for Geometrically Accurate Radiance Fields," 
        which introduces a novel approach to generating real-world quality meshes. Before diving into the review, 
        I present an impressive result from the Lego scene in the NeRFBlender dataset, created using 2D Gaussian Splatting (2D GS). It's quite something!
    </p>
</blockquote>

<figure>
    <img src="./240602_2dgs/assets/teaser.gif" alt="teaser" width="100%">
    <figcaption style="text-align: center; font-size: 15px;">
        TSDF reconstructed <strong>Mesh</strong> from 2D GS, made by hwan
    </figcaption>
</figure>

<br/>
<h3 id="recap-radiance-fields-mesh-recon-" class="lang eng"> Challenges in Mesh Reconstruction for Radiance Fields </h3>
<p class="lang eng"> 
    In a previous discussion (<span style="text-decoration: underline;"><a href="https://ncsoft.github.io/ncresearch/b515d0241ebe9af4a549e991ae0efc4a90f0f65e">Can NeRF be used in game production? (kor)</a></span>), 
    I highlighted the ongoing challenges in the practical application of Radiance Fields technologies. 
</p>
<div class="lang eng">
    <p>
        Although 3D Gaussian Splatting (3D GS) offers better portability to game and graphics engines compared to NeRF, 
        it still faces significant challenges in mesh reconstruction. 
    </p>
    <p>
        This difficulty arises due to the nature of 3D GS, which resembles a variant of point cloud representation, 
        making it inherently more complex to convert into a mesh than NeRF. 
    </p>
    <p>
        Recently, at SIGGRAPH 2024, "<a href="https://surfsplatting.github.io/">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</a>" 
        stands out as it demonstrates practical usability in mesh generation through a splatting-based approach. 
        Let’s take a look at what 2D GS is!
    </p>
</div>

<hr/>
<br/>

<h2 id="1-preliminary">1. Background </h2> <br/>
<h3 id="1-1-3d-gaussian-splatting"> 1.1. 3D Gaussian Splatting</h3>
<p><img src="./240602_2dgs/assets/image-7.png" alt="" width="100%"></p>

<div class="lang eng">
    <p>
        <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"> 3D Gaussian Splatting</a> is a technique that reconstructs 
        a 3D scene using a set of <em>anistropic &amp; explicit</em> primitive kernel.
    </p>
    <p>
        3D Gaussians. The original authors of 3D GS defined the covariance matrix of these 3D Gaussians as a density function 
        for a point $p$ in space, using the Gaussian rotation matrix $R$ and scale matrix $S$.
    </p>
</div>

<div class="math-container">
    $$ \begin{aligned}
    d(p) &= \sum_{g}
    \alpha_{g} \exp \left ( - \frac{1}{2} (p-\mu_{g})^{\rm T} \Sigma^{-1}_{g}  (p-\mu_{g})  \right )
    \\ \text{where } \mu &: \text{ center of gaussian, }
    \\ \Sigma&: \text{ cov mat, }   \textit{i.e., } RSS^{\rm T}R^{\rm T},
    \\ \alpha &: \text{ alpha-blending weight, } \textit{i.e.,} \text{ opacity value}
    \end{aligned}
    $$
</div>

<div class="lang eng">
    <p>
        3D GS, by definition, is similar to dense point cloud reconstruction but reconstructs 
        space as explicit radiance fields for novel view synthesis. 
        This explicit representation offers several advantages:
    </p>
    <ol>
        <li>
            <strong>Speed</strong><br/>
            Unlike NeRF, which requires querying a multi-layer perceptron (MLP) to obtain information, 
            3D GS stores the information of the 3D Gaussians explicitly, enabling real-time scene rendering at over 100 fps without the need to query an MLP.
        </li><br/>
        <li>
            <strong>Portability</strong> <br/>
            Since 3D GS requires only rasterization, it is much easier to port to game engines and implement in web viewers compared to NeRF.
            (<em>cf.</em> <a href="https://github.com/playcanvas/supersplat">SuperSplat</a>)
        </li><br/>
        <li>
            <strong>Editability</strong><br/>
            3D GS allows for straightforward scene editing, such as selecting, erasing, 
            or merging specific elements within a trained scene, which is more complex with NeRF due to its reliance on MLPs.
        </li>
    </ol>
</div>
<br/>

<h3 id="1-2-surface-reconstruction-problem-in-3d-gs"> 1.2. Surface Reconstruction Problem in 3D GS</h3>

<div class="lang eng">
    <p>
        Despite the advantages, 3D GS presents significant challenges in surface reconstruction. 
        The paper on 2D GS discusses four main reasons why surface reconstruction is difficult in 3D GS:
    </p>
    <ol>
        <li>
            <p>
                <strong>Difficulty in Learning Thin Surfaces</strong><br/>
                The volumetric radiance representation of 3D GS, which learns the three-dimensional scale, struggles to represent thin surfaces accurately.
            </p>
        </li>
        <li>
            <p>
                <strong>Absence of Surface Normals</strong><br/>
                Without surface normals, high-quality surface reconstruction is unattainable. 
                While INN addresses this with Signed Distance Functions (SDF), 3D GS lacks this feature.
            </p>
        </li>
        <li>
            <p>
                <strong>Lack of Multi-View Consistency</strong><br/>
                The rasterization process in 3D GS can lead to artifacts, as different viewpoints result in varying 2D intersection surfaces. <em>i.e.,</em> Artifacts! 
                <img src='./240602_2dgs/assets/image-8.png' width=40%>
            </p>
        </li>
        <li>
            <p>
                <strong>Inaccurate Affine Projection</strong><br/>
                The affine matrix used to convert 3D GS to radiance fields loses perspective accuracy as 
                it deviates from the Gaussian center, often leading to noisy reconstructions.
        </p>
        </li>
    </ol>
    <p>
        Additionally, 3D GS shares NeRF's challenge of generating high-quality meshes through methods 
        like Marching Cubes or Poisson Reconstruction, due to its volumetric opacity accumulation.
    </p>
</div>

<h3 id="1-3-sugar-surface-aligned-gaussian-splatting"> 1.3. SuGaR: Surface-Aligned Gaussian Splatting</h3>

<div class="lang eng">
    <p>
        <a href="https://anttwo.github.io/sugar/">SuGaR</a> (Surface-Aligned Gaussian Splatting) is a previous work that addresses some of the surface reconstruction challenges 
        in 3D GS. SuGaR's core idea is based on the assumption that for well-trained 3D Gaussians, <strong><em>the axis with the shortest scale will align with the surface normal</em></strong>. 
        This approximation is used as a regularization technique to ensure the 3D GS surface is aligned.
    </p>
</div>
<div class="math-container">
    $$ \begin{aligned}
    (p-\mu_{g^{\star}})^{\rm T} \Sigma^{-1}_{g^{\star}}  (p-\mu_{g^{\star}}) & \simeq \frac{1}{s_{g^{\star}}^{2}} \langle p-\mu_{g^{\star}}, n_{g^{\star}} \rangle ^{2}
    \\ \text{where } s_g &: \text{ smallest scaling factor},
    \\ n_g &: \text{ corresponding axis}
    \end{aligned} $$
</div>

<p class="lang eng">
    However, SuGaR is a two-stage method that first learns the 3D GS and then refines it, 
    leading to a complex learning process. Moreover, it does not fully resolve the projection inaccuracies that 
    contribute to surface reconstruction difficulties, often resulting in meshes with suboptimal geometry in custom scenes.
</p>

<h2 id="2-2d-gaussian-splatting">2. 2D Gaussian Splatting</h2><br/>
<h3 id="2-1-2d-gaussian-modeling-gaussian-surfels-"> 2.1. 2D Gaussian Modeling (Gaussian Surfels)</h3>
<p><img src ='./240602_2dgs/assets/image-6.png' width=100%></p>
<div class="lang eng">
    <p>
        The approach of 2D Gaussian Splatting (2D GS) essentially reverses the intuition behind SuGaR: 
        instead of flattening 3D Gaussians to align them with surfaces, 
        learns a scene composed of <em>flat</em> 2D Gaussians, known as surfels. 
    </p>
    <p>
        The Rotation Matrix $R$ and Scale Matrix $S$ for 2D GS can be defined accordingly.
    </p>
</div>

<div class="math-container">
    $$  \begin{aligned} 
    R &= [t_u, \ t_v, \ t_u \times t_v] \\ S &= [s_u, \ s_v, \ 0] 
    \end{aligned}
    $$
</div>

<p class="lang eng">
    In this context, a 2D Gaussian is defined as a local tangent plane $P$ in the $uv$ coordinate frame 
    which has center point $p_k$, tanget vector $(t_u, t_v)$.
</p>
<div class="math-container">
    $$ \begin{aligned} 
    P(u,v) &= p_k + s_ut_u u + s_v t_v v = \mathbf{H}(u,v,1,1)^{\rm T} \\ 
    \text{where } \mathbf{H} &= 
    \begin{bmatrix} 
    s_u t_u & s_v t_v & 0 & p_k \\ 
    0 & 0 & 0& 1 \\
    \end{bmatrix} =
    \begin{bmatrix} 
    RS &  p_k \\ 
    0 & 1
    \end{bmatrix}
    \end{aligned}
    $$
</div>

<p class="lang eng">
    Therefore, the 2D Gaussian is represented by a standard normal function.
</p>
<div class="math-container">
    $$ \mathcal{G}(u,v) = \exp \left( -\frac{u^2 + v^2}{2} \right)
    $$ 
</div>

<p class="lang eng">
    The primary parameters to be learned in 2D GS include the rotation axis, scaling, 
    and spherical harmonics coefficients for opacity and non-Lambertian color.
</p>


<h3 id="2-2-splatting"> 2.2. Splatting </h3><br/>
<h4 id="2-2-1-accurate-2d-to-2d-projection-in-homogeneous-coordinates"> 2D-to-2D Projection </h4>

<div class="lang eng">
    <p>
        In principle, a 2D Gaussian can be used as a 3D GS projection by simply setting the third scale dimension to zero. 
        However, the affine projection method used in 3D GS, based on the first-order Taylor expansion $\Sigma&#39; = JW\Sigma W^{\rm T}J^{\rm T}$, 
        induces approximation errors as the distance from the Gaussian center increases.
    </p>
    <blockquote>
        related discussion in official repo:
    </blockquote>
</div>
<img src="./240602_2dgs/assets/image-5.png" alt="" width="90%"></li>

<div class="lang eng">
    <p>
        To address these inaccuracies, the authors of 2D GS propose using a conventional 2D-to-2D mapping in <strong><em>homogeneous coordinates</em></strong>. 
    </p>
    <p> 
        For a world-to-screen transformation matrix $\mathbf{W} \in \mathbb{R}^{4 \times 4}$, 
        a 2D point $(x,y)$ in screen space can be derived as follows:
    </p>
</div>
<div class="math-container">
    $$ \mathbf{x} = (xz, yz, z, z)^{\rm T} = \mathbf{W} P(u, v) = \mathbf{WH}(u,v,1,1)^{\rm T}
    $$
</div>
<p class="lang eng">
    This means that the c2w direction ray from a point in camera space intersects the 2D splats at depth $z$, 
    and the Gaussian density of the screen space point can be obtained as
</p>
<div class="math-container">
    $$ \mathcal{G} \left( (\mathbf{WH})^{-1} \mathbf{x} \right)
    $$
</div>
<p class="lang eng">
    However, the inverse transform can introduce numerical instabilities, leading to optimization challenges.
</p>

<h4 id="2-2-2-ray-splat-intersection"> Ray-Splat Intersection w/ Homography</h4>

<div class="lang eng">
    <p>
        The authors resolve the ray-splat intersection problem by identifying the intersection of three non-parallel planes. 
    </p>
    <p>
        For a given image coordinate $(x, y)$, a ray is defined as the intersection between two homogeneous planes, 
        $\mathbf{h}_x = (-1, 0, 0, x)$ and $y$-plane $\mathbf{h}_y = (0, -1, 0, y)$.
        And the intersection is calculated by transforming the homogeneous planes $\mathbf{h}_x$, $\mathbf{h}_y$ 
        to $uv$-space.
    </p>
</div>
<div class="math-container">
    $$ \mathbf{h}_u = (\mathbf{WH})^{\rm T}\mathbf{h}_x,  \quad \mathbf{h}_v = (\mathbf{WH})^{\rm T}\mathbf{h}_y
    $$
</div>

<p class="lang eng">
    By homography, the two planes $\mathbf{h}_u$ and $\mathbf{h}_v$ in $uv$-space are used to find the intersection point 
    with the 2D Gaussian splats. 
</p>
<div class="math-container">
    $$ \mathbf{h}_u \cdot (u,v,1,1)^{\rm T} = \mathbf{h}_v \cdot (u,v,1,1)^{\rm T} = 0 , $$
</div>
<div class="math-container">
    $$ 
    u(\mathbf{x}) = \frac{\mathbf{h}_u^2 \mathbf{h}_v^4 - \mathbf{h}_u^4 \mathbf{h}_v^2}{\mathbf{h}_u^1 \mathbf{h}_v^2 - \mathbf{h}_u^2 \mathbf{h}_v^1} , \quad
    v(\mathbf{x}) = \frac{\mathbf{h}_u^4 \mathbf{h}_v^1 - \mathbf{h}_u^1 \mathbf{h}_v^4}{\mathbf{h}_u^1 \mathbf{h}_v^2 - \mathbf{h}_u^2 \mathbf{h}_v^1}
    $$
</div>
<div class="lang eng">
    <p>
        This closed-form solution provides the projection value from the $uv$-space for the screen pixel, 
        with the depth $z$ obtained from a previously defined equation.
    </p>
    <p>
        Comparative figures in the supplementary material of the paper demonstrate that the 
        homogeneous projection method in 2D GS is more accurate than the affine projection used in 3D GS.
    </p>
</div>
<img src='./240602_2dgs/assets/image-2.jpeg' width="100%">
<br/>
<br/>


<h3 id="2-3-training-2d-gs"> 2.3. Training 2D GS </h3>
<
<p class="lang eng">
    In addition to the image rendering loss defined through 2D projection and rasterization, 
    the training process for 2D GS incorporates two additional regularization losses: Depth Distortion Loss and Normal Consistency Loss.
</p>
<h4 id="2-3-1-depth-distortion"> Depth Distortion Regularization </h4>

<div class="lang eng">
    <p>
        Unlike NeRF, where volume rendering accounts for distance differences between intersecting splats, 3D GS does not, 
        leading to challenges in surface reconstruction. 
    </p>
    <p>
        To address this, 
        the authors introduce a depth distortion loss, which concentrates the ray weight distribution near the ray-splat intersection, 
        similar to Mip-NeRF360.
    </p>
</div>
<div class="math-container">
    $$ 
    L_d = \sum_{i,j} \omega_i \omega_j |z_i - z_j|
    $$
</div>
<p class="lang eng">
    Here, for ray-splat inersection, each variable represents followings:
</p>
<ul>
    <li>
        $\omega_i = \alpha_i \hat{G}_i(u(x)) \prod_{j=1}^{i-1}(1 - \alpha_j \hat{G}_j(u(x)))$ : blending weight
    </li>
    <li>
        $z_i$: depth 
    </li>
</ul>
<div class="lang eng">
    <p>
        The definition of weight shows that it is the same expression as the accumulated transmittance of the NeRF, and by the same logic, 
        when a point is transparent along the current ray direction and the opacity value of the point is high, the weight will be a large value. 
    </p>
    <p>
        In other words, the loss is a regularization that reduces the depth difference between ray-splat intersections with high opacity.
    </p>
</div>

<div class="math-container">
    $$
    \begin{aligned}
    \mathcal{L} 
    &= \sum_{i=0}^{N-1} \sum_{j=0}^{i-1} \omega_i \omega_j (m_i - m_j)^2 \\
    &= \sum_{i=0}^{N-1} \omega_i \left( m_i^2 \sum_{j=0}^{i-1} \omega_j + \sum_{j=0}^{i-1} \omega_j m_j^2 - 2m_i \sum_{j=0}^{i-1} \omega_j m_j \right) \\
    &= \sum_{i=0}^{N-1} \omega_i \left( m_i^2 A_{i-1} + D_{i-1}^2 - 2m_i D_{i-1} \right),
    \end{aligned}
    $$
</div>
<p> 
    where $A_i = \sum_{j=0}^{i} \omega_j$, $D_i = \sum_{j=0}^{i} \omega_j m_j$, and $D_i^2 = \sum_{j=0}^{i} \omega_j m_j^2$
</p>
<p class="lang eng">
    Also, the implementation of this loss shows that, 
    each of the terms is either an accumulation of opacity or an accumulation of opacity $\times$ depth.
</p>

<div class="math-container">
    $$ 
    \begin{aligned}
    \mathcal{L}_i &= \sum_{j=0}^{i} \omega_j e_j \\
    \text{where } e_i &= m_i^2 A_{i-1} + D_{i-1}^2 - 2m_i D_{i-1}
    \end{aligned}
    $$
</div>
<p class="lang eng">
    Therefore, this loss can be computed as if it were an image rendering, using the above formula. 
    In the 2D GS implementations, this is handled by the rasterizer.
</p>

<h4 id="2-3-2-normal-consistency"> Normal Consistency Regularization </h4>

<div class="lang eng">
    <p>
        In addition to distortion loss, 2D GS presents normal consistency loss, 
        which ensures that all 2D splats are aligned with the real surface.
    </p>
    <p>
        Since Volume Rendering allows for multiple translucent 2D Gaussians (surfels) to exist along a ray, 
        the authors consider the area where the accumulated opacity reaches 0.5 to be the true surface.
    </p>
    <p>
        They propose a normal consistency loss that aligns the derivative of the surface's normal and depth in this region as follows:
    </p>
</div>
<div class="math-container">
    $$ 
    L_n = \sum_{i} \omega_i (1 - \mathbf{n}_i^\top \mathbf{N})
    $$
</div>
<div class="lang eng">
    <p> where </p>
    <ul>
        <li>$i$ is the index of the splats intersecting along the ray</li>
        <li>$\omega_i$ is the blending weight of ray-splat intersections</li>
        <li>$\mathbf{n}_i$ is the normal vector of splats</li>
        <li>$\mathbf{N}$ is the normal vector estimated from points in the neighboring depth map. </li>
    </ul>
    <p> Specifically, $\mathbf{N}$ is computed using finite difference as follows. </p>
</div>
<div class="lang kor" style="display: none;">
    <p> 여기서, </p>
    <ul>
        <li>$i$ 는 ray 을 따라 교차하는 splats 의 index</li>
        <li>$\omega_i$ 는 ray-splat 교점의 blending weight</li>
        <li>$\mathbf{n}_i$ 는 splat 의 normal vector</li>
        <li>$\mathbf{N}$ 은 인근 depth map 의 point $\mathbf{p}$ 에서 추정된 normal vector 이다 </li>
    </ul>
    <p>구체적으로, $\mathbf{N}$은 finite difference 을 사용하여 다음과 같이 계산된다. </p>
</div>
<div class="math-container">
    $$
    \mathbf{N} (x, y) = \frac{ \nabla _x \mathbf{p}  \times \nabla_y \mathbf{p}}{| \nabla_x \mathbf{p} \times \nabla_y \mathbf{p} | }
    $$
</div>
<p class="lang eng">
    This loss aligns the derivative of the surface's normal and depth in this region, 
    ensuring consistency between the normal vector of splats and the normal vector estimated from the neighboring depth map.
</p>
<figure>
    <img src="./240602_2dgs/assets/image-3.png" alt="alt description" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure:</strong> Normal vs Depth2Normal, captured in my custom viewer </figcaption>
</figure>
<br/>

<h2 id="3-experimens-custom-viser-viewer">3. Experimens &amp; Custom Viewer</h2><br/>
<h3 id="3-1-qualitative-results-custom-object-reconstruction">3.1. Qualitative Results </h3>

<p class="lang eng">
    Although the quantitative results on Novel view synthesis may not appear impressive in comparison 
    to previous studies, the true value of 2D GS lies in its qualitative evaluation and practical testing. 
    The performance of 2D GS in surface and mesh reconstruction surpasses that of any previous work.
</p>
<p><img src="./240602_2dgs/assets/image.png" alt="" width="100%"></p>

<div class="lang eng">
    <p>
        This high performance extends to real-world custom scenes. 
        Experimental results on two custom objects, a guitar, and a penguin.
    </p>
</div>
<ul>
    <li><p><strong>2D GS: guitar (mesh) </strong>
    <img src="./240602_2dgs/assets/image.gif" alt="" width="100%"></p>
    </li>
    <li><p><strong>2D GS: penguin (mesh) </strong>
    <img src="./240602_2dgs/assets/image-4.gif" alt="" width="100%"></p>
    </li>
</ul>

<div class="lang eng">
    <p>
        It demonstrates that 2D GS can generate meshes of sufficient quality for use in games and modeling, provided that issues related to light condition disentanglement are addressed.
    </p>
    <p>
        Moreover, 2D GS's ability to accurately extract depth values enables quick mesh generation through Truncated Signed Distance Function (TSDF) reconstruction, with our experiments completing the process in under a couple of minutes.
    </p>
    <blockquote>
        <p>
            <em>cf.</em> 
            <a href="https://turandai.github.io/projects/gaussian_surfels/">Gaussian Surfels</a> , 
            which was presented at SIGGRAPH'24 uses the same idea. 
            However, instead of using the 3rd axis as a cross-product of the 1st and 2nd axes, 
            it learns the 3rd axis separately but uses the rasterization of the original 3D GS with scale equal to 0. 
            This means that it does not solve the affine projection error. 
            When we tested the algorithm in practice, we found that 2D GS outperformed Gaussian Surfels.
        </p>
    </blockquote>
</div>


<h3 id="3-2-custom-viser-viewer-for-2d-gaussian-splatting">3.2. Custom Viser Viewer for 2D Gaussian Splatting</h3>

<div class="lang eng">
    <p>
        <del>One limitation of 2D GS is the lack of an official viewer. </del>
        (as of 24.06.10, SIBR Viewer is available).
    </p>
    <p>
        If you add scale dimension to the 2D GS ply file and assign its value to 0, 
        you can still use the 3D GS viewer, but you will not be able to use an accurate Gaussian projection, 
        which is one of the main contributions of the 2D GS authors.
    </p>
    <p>
        To overcome this limitation, I developed a custom viewer using Viser, 
        which supports the homogeneous projection of 2D GS, eliminating projection errors. 
        This viewer offers various visualization and editing functions, 
        making it easier to monitor scene training and generate rendering camera paths.
    </p>
</div>
<h4 id="-github-project-link-https-github-com-hwanhuh-2d-gs-viser-viewer-tree-main-">⭐ <a href="https://github.com/hwanhuh/2D-GS-Viser-Viewer/tree/main">Github Project Link</a></h4>
<p><img src="./240602_2dgs/assets/viser_train.gif" alt="" width="100%"></p>

<br/>

<h2 id="4-conclusion">4. Conclusion</h2>

<div class="lang eng">
    <p>
        In conclusion, the introduction of 2D Gaussian Splatting marks a significant advancement in the practical use 
        of Radiance Fields. 
    </p>
    <p>
        Building upon the explicit representation of 3D GS, 2D GS not only addresses previous challenges in projection inaccuracies 
        and rasterization but also demonstrates superior performance in both synthetic and real-world scenarios. 
        This algorithm represents a well-designed approach to mesh generation, with promising applications in games 
        and modeling.
    </p>
</div>
<hr/>
<p>
    You may also like, 
</p>
<ul>
    <li>
        <a href="./?id=240805_gs/">
            <span style="text-decoration: underline;">A Comprehensive Analysis of Gaussian Splatting Rasterization</span>
        </a>
    </li>
    <li>
        <a href="./?id=240823_grt/">
            <span style="text-decoration: underline;">Don't Rasterize, But Ray Trace 3D Gaussian</span>
        </a>
    </li>
</ul>
<br/>