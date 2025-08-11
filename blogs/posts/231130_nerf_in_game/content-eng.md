title: Can NeRF be Used in Game?
date: November 30, 2024
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
        <li><a href="#introduction"> Introduction</a></li>
        <li><a href="#main"> NeRF & Illumination Control </a></li>
        <ul>
            <li><a href="#problem"> NeRF&#39;s Inability to Separate Lighting Effects </a></li>
            <li><a href="#separating-lighting-effects-from-nerf"> Separating Lighting Effects from NeRF </a></li>
        </ul>
        <li><a href="#challenges"> Other Challenges and Solutions </a></li>
        <li><a href="#nerf_in_game"> Neural Rendering in Game </a></li>
    </ul>
</nav>

<br/>
<h2 id="introduction">1. Introduction</h2>
<p>In this article, we&#39;ll explore some of the limitations of using NeRF (Neural Radiance Fields) in game production and how to work around them.</p>
<ul>
    <li>
        <p>
            kor version: <span style="text-decoration: underline;"><a href="https://ncsoft.github.io/ncresearch/b515d0241ebe9af4a549e991ae0efc4a90f0f65e">link</a></span>
        </p>
    </li>
</ul>

<h3 id="can-nerf-be-used-in-game-production-">Can NeRF be Used in Game Production?</h3>
<p>As discussed in &quot;Creating Realistic 3D Models with NeRF,&quot; NeRF is a technique for reconstructing 3D models from a set of images taken from various angles. With the ability to easily obtain high-quality 3D models, NeRF appears to be a promising technology for game development. But are NeRF models ready to be integrated into game production?</p>
<figure>
    <img src="./231130_nerf_in_game/assets/diget.gif" alt="Gaussian RT" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 1.</strong> NeRF Reconstruction of Cookie Box </figcaption>
</figure>
<p>Can we use the 3D model of a cookie box, as an object in a game? Unfortunately, the answer is still &#39;<strong><em>No</em></strong>&#39;. </p>
<p>Several limitations need to be addressed before NeRF can be used directly in commercial applications or game production. In this post, I&#39;ll outline the primary obstacles preventing NeRF&#39;s direct application in game development and the solutions being developed to overcome them.</p>

<h2 id="main"> 2. NeRF & Illumination Control </h2><br/>
<h3 id="problem">NeRF&#39;s Inability to Separate Lighting Effects</h3>
<figure>
    <img src="./231130_nerf_in_game/assets/nerf.gif" alt="Gaussian RT" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 2.</strong> NeRF reconstructed Objects </figcaption>
</figure>
<p>The image above shows a set of 3D objects generated with NeRF. While the models are of high quality, there is a significant limitation: <br/> <em> The shadows are fixed.</em> </p>
<p style="text-align: center;"><em><strong>Why is this problematic?</strong></em> </p>
<p>Imagine rotating an object in your space—the direction of the shadow would change with the object&#39;s movement, right? However, in the case of a NeRF model, the light and object remain static, meaning the shadow stays consistent with what an observer would see when moving around the object.</p>
<p>In other words, NeRF models cannot separate the reflections of the captured light from the object itself. This is a critical flaw, especially considering that Physics-Based Rendering (PBR), essential for creating realistic games, relies on realistic lighting effects.</p>
<p>Even in Phong shading, one of the simplest forms of light shading, the reflection effects are handled by three components:</p>
<figure>
    <img src="./231130_nerf_in_game/assets/IMG_2_Phong_reflection_model.jpg" alt="Gaussian RT" width="100%"> 
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 3.</strong> Phong reflection model, source: <span style="text-decoration: underline;"><a href="https://learnopengl.com/Lighting/Basic-Lighting">OpenGL</a></span> </figcaption>
</figure>
<ul>
<li><strong>Ambient</strong>: Represents the ambient light.</li>
<li><strong>Diffuse</strong>: Represents the scattered light.</li>
<li><strong>Specular</strong>: Represents the shiny reflections.</li>
</ul>
<p>A model trained with NeRF cannot separate these lighting effects, resulting in a 3D model where the captured lighting conditions are baked into the object. To apply even basic relighting, we need to separate at least the diffuse color, which closely resembles the intrinsic color of the object.</p>

<h3 id="separating-lighting-effects-from-nerf">Separating Lighting Effects from NeRF</h3>
<p>So, how can we separate lighting effects in NeRF models? There are several methods, but one simple approach involves modifying the model architecture to output specular and diffuse components separately. </p>
<p>Unlike the standard NeRF network structure, which takes coordinates $x$ (spatial location) and $d$ (direction) as inputs and outputs an RGB color $c$, a modified NeRF model outputs both specular color $c_s$ and diffuse color $c_d$.</p>
<figure>
    <img src="./231130_nerf_in_game/assets/IMG_3_diffuse_specular_disentanglement_model.jpg" alt="Gaussian RT" width="100%"> 
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 4.</strong> NeRF MLP w/ Diffuse-Specular Disentanglement </figcaption>
</figure>
<p>In this structure, the Spatial MLP not only outputs the diffuse color $c_d$ but also the normal vector $n$ at that location. This normal vector is used to represent the reflected light more accurately through a technique called &#39;reflection reparameterization.&#39; Consequently, the reflected light parameter $w_r$ is input into the Directional MLP, which then outputs the specular color $c_s$.</p>
<p>Understanding the reflection reparameterization formula might not be straightforward initially, but consider the following scenario to understand why we can represent reflected light $w_r$ through the normal vector and the observation direction.</p>
<figure>
    <img src="./231130_nerf_in_game/assets/IMG_4_reflection_reparameterization.jpg" alt="Gaussian RT" width="100%"> 
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 5.</strong> Illustration of reflection parameterization </figcaption>
</figure>
<p>As illustrated above, the reflected light $w_r$ at a point on an object has the following relationship with the observer&#39;s direction $d$ and normal $n$:</p>
<p>
$$
\frac{(w_r + d)}{2} = (d \cdot n) \cdot n
$$
</p>
<p>By slightly modifying this relationship, we can derive the reflection reparameterization used earlier. Below is an example of how the lighting effects appear using this method.</p>
<table>
    <tr>
        <th>Full</th>
        <th>Diffuse</th>
        <th>Specular</th>
    </tr>
</table>
<div class="video-container" style="padding-top:2%; padding-bottom:28%">
    <video controls style="width: 100%">
        <source src="./231130_nerf_in_game/assets/VID_2_illumination_control_scene_watermark.mp4" type="video/mp4">
    </video>
</div>
<table>
    <tr>
        <th>Full</th>
        <th>Diffuse</th>
        <th>Specular</th>
    </tr>
</table>
<div class="video-container" style="padding-top:2%; padding-bottom:28%">
    <video controls style="width: 100%">
        <source src="./231130_nerf_in_game/assets/VID_3_illumination_control_object_watermark.mp4" type="video/mp4">
    </video>
</div>
<p>While this does not achieve perfect diffuse-specular separation, it shows that the object&#39;s color can be modeled independently of the lighting effects to some extent, compared to the full NeRF model on the left. Using only the diffuse NeRF model in the middle allows for relighting to be applied.</p>
<br/>
<h2 id="challenges">3. Other Challenges and Solutions</h2><br/>
<h3 id="slow">Slow Rendering Speed</h3>
<p>
    NeRF stores information in a 3D space through a neural network, meaning that to retrieve scene information, we must pass it through the network. Regardless of how lightweight the network is, the speed difference between directly reading stored information and obtaining it through the network is significant. This results in slow rendering speeds, a major drawback of NeRF.
</p>
<p>However, this limitation has been largely mitigated with methods like Instant-NGP, Plenoxels, TriMipRF, and Zip-NeRF, which store features in select 3D locations and use interpolation to reduce computational load. </p>

<h3 id="engine">Game Engine Compatibility</h3>
<p>
    Since Neural Rendering is not compatible with the conventional renderers used by game or graphics engines, a custom renderer must be created to use NeRF. However, the slow rendering speed of NeRF, as mentioned earlier, presents a significant barrier to practical use. Therefore, in most cases, objects created with NeRF should be converted to explicit meshes using methods like Marching Cubes.
</p>
<p>Due to the semi-transparent nature of NeRF opacity, creating high-quality meshes is challenging. However, new meshing techniques for neural rendering, such as nerf2mesh and flexicubes, are emerging to address this issue.</p>
<p>Recently, Gaussian Splatting, which uses explicit radiance fields and rasterization, has also been introduced. Gaussian Splatting is naturally compatible with game engines and is rapidly evolving.</p>

<h2 id="nerf_in_game">4. Neural Rendering in Game Engines</h2>
<p>In the early stage of Neural Rendering, neural rendering technologies faced several limitations that hindered their use in-game production and other applications. </p>
<p>However, as these challenges are gradually being resolved, novel methods are emerging that enable neural rendering technologies in commercial game engines. Technology startups focusing on this technology are also appearing.</p>
<div class="video-container">
    <video controls style="width: 100%">
        <source src="./231130_nerf_in_game/assets/VID_4_NeRF_obeject_in_blender.mp4" type="video/mp4">
    </video>
</div>
<p>The video above demonstrates the use of NeRF-restored objects alongside other 3D objects in graphics software. In the video, the table is an object restored through NeRF, while the dinosaur and butterfly are predefined presets.</p>
<p>Additionally, NeRF and neural rendering technologies can be used not only for restoring and utilizing individual objects but also for reconstructing large real-world spaces to serve as maps in games.</p>
<figure>
    <img src="./231130_nerf_in_game/assets/IMG_5_360_camera_capture.jpg" alt="Gaussian RT" width="100%"> 
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 6.</strong> 360 Camera Capture </figcaption>
</figure>
<div class="video-container">
    <iframe width="100%" src="https://www.youtube.com/embed/ISm-IL3HzmM?si=tX3dQJXe7fn8uJv0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

<div class="video-container">
    <iframe width="100%" src="https://www.youtube.com/embed/3OqbvUaoNFw?si=Yo8XNgqaWnna4zS_" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>
<p>The video above shows an example of capturing the NCSOFT backyard park, reconstructing it into a 3D space, and using it in a game. </p>
<p>To efficiently capture the space, a special camera with a wide field of view was used to scan the entire 360-degree area, as shown in the video, and then reconstructed using neural rendering technology. </p>
<p>Not only does it replicate the appearance exactly as photographed, but it can also be used as a game environment with a unique atmosphere, as seen in the latter part of the video. I&#39;m looking forward to seeing many exciting games based on neural rendering technology when combined with innovative ideas.</p>

<p>
    If you are more interested, please refer to my project: <span style="text-decoration: underline;"><a href="../../../projects/nerf_in_game/">Neural Rendeing in Game Engine</a></span>
</p>
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