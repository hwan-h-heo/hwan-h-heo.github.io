title: Can NeRF be Used in Game?
date: November 30, 2024
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

## <span id="introduction"></span>1. Introduction

In this article, we&#39;ll explore some of the limitations of using NeRF (Neural Radiance Fields) in game production and how to work around them.

<ul>
    <li>
        <p>
            kor version: <span style="text-decoration: underline;"><a href="https://ncsoft.github.io/ncresearch/b515d0241ebe9af4a549e991ae0efc4a90f0f65e">link</a></span>
        </p>
    </li>
</ul>

### <span id="can-nerf-be-used-in-game-production-"></span>Can NeRF be Used in Game Production?

As discussed in &quot;Creating Realistic 3D Models with NeRF,&quot; NeRF is a technique for reconstructing 3D models from a set of images taken from various angles. With the ability to easily obtain high-quality 3D models, NeRF appears to be a promising technology for game development. But are NeRF models ready to be integrated into game production?

<figure>
    <video class="post-media" autoplay loop muted playsinline preload="metadata" poster="./231130_nerf_in_game/assets/diget-poster.jpg" aria-label="Gaussian RT" style="width: 100%">
        <source src="./231130_nerf_in_game/assets/diget.mp4" type="video/mp4">
    </video>
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 1.</strong> NeRF Reconstruction of Cookie Box </figcaption>
</figure>

Can we use the 3D model of a cookie box, as an object in a game? Unfortunately, the answer is still &#39;**_No_**&#39;.

Several limitations need to be addressed before NeRF can be used directly in commercial applications or game production. In this post, I&#39;ll outline the primary obstacles preventing NeRF&#39;s direct application in game development and the solutions being developed to overcome them.

## <span id="main"></span>2. NeRF & Illumination Control

### <span id="problem"></span>NeRF&#39;s Inability to Separate Lighting Effects

<figure>
    <video class="post-media" autoplay loop muted playsinline preload="metadata" poster="./231130_nerf_in_game/assets/nerf-poster.jpg" aria-label="Gaussian RT" style="width: 100%">
        <source src="./231130_nerf_in_game/assets/nerf.mp4" type="video/mp4">
    </video>
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 2.</strong> NeRF reconstructed Objects </figcaption>
</figure>

The image above shows a set of 3D objects generated with NeRF. While the models are of high quality, there is a significant limitation: <br/> <em> The shadows are fixed.</em>

<p style="text-align: center;"><em><strong>Why is this problematic?</strong></em> </p>

Imagine rotating an object in your space—the direction of the shadow would change with the object&#39;s movement, right? However, in the case of a NeRF model, the light and object remain static, meaning the shadow stays consistent with what an observer would see when moving around the object.

In other words, NeRF models cannot separate the reflections of the captured light from the object itself. This is a critical flaw, especially considering that Physics-Based Rendering (PBR), essential for creating realistic games, relies on realistic lighting effects.

Even in Phong shading, one of the simplest forms of light shading, the reflection effects are handled by three components:

<figure>
    <img src="./231130_nerf_in_game/assets/IMG_2_Phong_reflection_model.jpg" alt="Gaussian RT" width="100%"> 
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 3.</strong> Phong reflection model, source: <span style="text-decoration: underline;"><a href="https://learnopengl.com/Lighting/Basic-Lighting">OpenGL</a></span> </figcaption>
</figure>
<ul>
<li><strong>Ambient</strong>: Represents the ambient light.</li>
<li><strong>Diffuse</strong>: Represents the scattered light.</li>
<li><strong>Specular</strong>: Represents the shiny reflections.</li>
</ul>

A model trained with NeRF cannot separate these lighting effects, resulting in a 3D model where the captured lighting conditions are baked into the object. To apply even basic relighting, we need to separate at least the diffuse color, which closely resembles the intrinsic color of the object.

### <span id="separating-lighting-effects-from-nerf"></span>Separating Lighting Effects from NeRF

So, how can we separate lighting effects in NeRF models? There are several methods, but one simple approach involves modifying the model architecture to output specular and diffuse components separately.

Unlike the standard NeRF network structure, which takes coordinates $x$ (spatial location) and $d$ (direction) as inputs and outputs an RGB color $c$, a modified NeRF model outputs both specular color $c_s$ and diffuse color $c_d$.

<figure>
    <img src="./231130_nerf_in_game/assets/IMG_3_diffuse_specular_disentanglement_model.jpg" alt="Gaussian RT" width="100%"> 
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 4.</strong> NeRF MLP w/ Diffuse-Specular Disentanglement </figcaption>
</figure>

In this structure, the Spatial MLP not only outputs the diffuse color $c_d$ but also the normal vector $n$ at that location. This normal vector is used to represent the reflected light more accurately through a technique called &#39;reflection reparameterization.&#39; Consequently, the reflected light parameter $w_r$ is input into the Directional MLP, which then outputs the specular color $c_s$.

Understanding the reflection reparameterization formula might not be straightforward initially, but consider the following scenario to understand why we can represent reflected light $w_r$ through the normal vector and the observation direction.

<figure>
    <img src="./231130_nerf_in_game/assets/IMG_4_reflection_reparameterization.jpg" alt="Gaussian RT" width="100%"> 
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 5.</strong> Illustration of reflection parameterization </figcaption>
</figure>

As illustrated above, the reflected light $w_r$ at a point on an object has the following relationship with the observer&#39;s direction $d$ and normal $n$:

$$
\frac{(w_r + d)}{2} = (d \cdot n) \cdot n
$$

By slightly modifying this relationship, we can derive the reflection reparameterization used earlier. Below is an example of how the lighting effects appear using this method.

<table>
    <tr>
        <th>Full</th>
        <th>Diffuse</th>
        <th>Specular</th>
    </tr>
</table>
<video class="post-media" controls preload="metadata" playsinline style="width: 100%">
    <source src="./231130_nerf_in_game/assets/VID_2_illumination_control_scene_watermark.mp4" type="video/mp4">
</video>
<table>
    <tr>
        <th>Full</th>
        <th>Diffuse</th>
        <th>Specular</th>
    </tr>
</table>
<video class="post-media" controls preload="metadata" playsinline style="width: 100%">
    <source src="./231130_nerf_in_game/assets/VID_3_illumination_control_object_watermark.mp4" type="video/mp4">
</video>

While this does not achieve perfect diffuse-specular separation, it shows that the object&#39;s color can be modeled independently of the lighting effects to some extent, compared to the full NeRF model on the left. Using only the diffuse NeRF model in the middle allows for relighting to be applied.

## <span id="challenges"></span>3. Other Challenges and Solutions

### <span id="slow"></span>Slow Rendering Speed

NeRF stores information in a 3D space through a neural network, meaning that to retrieve scene information, we must pass it through the network. Regardless of how lightweight the network is, the speed difference between directly reading stored information and obtaining it through the network is significant. This results in slow rendering speeds, a major drawback of NeRF.

However, this limitation has been largely mitigated with methods like Instant-NGP, Plenoxels, TriMipRF, and Zip-NeRF, which store features in select 3D locations and use interpolation to reduce computational load.

### <span id="engine"></span>Game Engine Compatibility

Since Neural Rendering is not compatible with the conventional renderers used by game or graphics engines, a custom renderer must be created to use NeRF. However, the slow rendering speed of NeRF, as mentioned earlier, presents a significant barrier to practical use. Therefore, in most cases, objects created with NeRF should be converted to explicit meshes using methods like Marching Cubes.

Due to the semi-transparent nature of NeRF opacity, creating high-quality meshes is challenging. However, new meshing techniques for neural rendering, such as nerf2mesh and flexicubes, are emerging to address this issue.

Recently, Gaussian Splatting, which uses explicit radiance fields and rasterization, has also been introduced. Gaussian Splatting is naturally compatible with game engines and is rapidly evolving.

## <span id="nerf_in_game"></span>4. Neural Rendering in Game Engines

In the early stage of Neural Rendering, neural rendering technologies faced several limitations that hindered their use in-game production and other applications.

However, as these challenges are gradually being resolved, novel methods are emerging that enable neural rendering technologies in commercial game engines. Technology startups focusing on this technology are also appearing.

<video class="post-media" controls preload="metadata" playsinline poster="./231130_nerf_in_game/assets/VID_4_NeRF_obeject_in_blender-poster.jpg" style="width: 100%">
    <source src="./231130_nerf_in_game/assets/VID_4_NeRF_obeject_in_blender.mp4" type="video/mp4">
</video>

The video above demonstrates the use of NeRF-restored objects alongside other 3D objects in graphics software. In the video, the table is an object restored through NeRF, while the dinosaur and butterfly are predefined presets.

Additionally, NeRF and neural rendering technologies can be used not only for restoring and utilizing individual objects but also for reconstructing large real-world spaces to serve as maps in games.

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

The video above shows an example of capturing the NCSOFT backyard park, reconstructing it into a 3D space, and using it in a game.

To efficiently capture the space, a special camera with a wide field of view was used to scan the entire 360-degree area, as shown in the video, and then reconstructed using neural rendering technology.

Not only does it replicate the appearance exactly as photographed, but it can also be used as a game environment with a unique atmosphere, as seen in the latter part of the video. I&#39;m looking forward to seeing many exciting games based on neural rendering technology when combined with innovative ideas.

If you are more interested, please refer to my project: <span style="text-decoration: underline;">[Neural Rendeing in Game Engine](../../../projects/nerf_in_game/)</span>