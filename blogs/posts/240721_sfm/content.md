title: Radiance Fields from Deep-based Structure-from-Motion
date: July 21, 2024
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
        <ul>
            <li><a href="#sfm"> What is SfM?</a></li>
            <li><a href="#colmap"> about COLMAP</a></li>
        </ul>
        <li>
            <a href="#sec2"> Deep Learning-Based Camera Pose Reconstruction </a>
        </li>
        <ul>
            <li><a href="#sec2.1">VGGSfM</a></li>
            <li><a href="#sec2.2"> MASt3R</a></li>
        </ul>
        <li><a href="#sec3"> Radiance Fields from Deep-based SfM </a></li>
        <li>
            <a href="#closing"> Closing </a>
        </li>
    </ul>
</nav>

<br/>
<h2 id="intro"> 1. Introduction</h2>
<br/>
<h3 id="sfm"> What is Structure-from-Motion? </h3>
<figure>
    <img src="./240721_sfm/assets/sfm.jpg" alt="Structure-from-Motion" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 1.</strong> Structure-from-Motion, source: Privacy Preserving Structure-from-Motion</figcaption>
</figure>
<p>
    Structure from Motion (SfM) is a computer vision technique that reconstructs 3D structures 
    from 2D images. It's widely used in various fields such as drone mapping, robotics, 
    and virtual reality. The primary goal of SfM is to estimate both the camera positions and 
    the 3D point cloud from multiple overlapping images.
</p>

<h3 id="colmap">COLMAP: The Gold Standard in SfM</h3>
<br/>
<figure>
    <img src="./240721_sfm/assets/colmap.jpg" alt="colmap" width="100%">
    <br/>
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 2.</strong> COLMAP, source: <span style="text-decoration: underline;"><a href="https://colmap.github.io/">COLMAP</a></span></figcaption>
</figure>
<p>
    COLMAP is a widely recognized software for Structure from Motion (SfM) and Multi-View Stereo (MVS), 
    playing a crucial role in the 3D reconstruction community. 
</p>
<p>
    Developed by Johannes L. Schönberger, COLMAP has become the gold standard 
    for 3D reconstruction tasks due to its robustness, accuracy, and versatility.
    It offers several robust features:
</p>
<h4>How COLMAP Works</h4>
<br/>
<img src="./240721_sfm/assets/colmap_pipe.jpg" alt="colmap pipeline" width="100%">
<br/>
<br/>
<ol>
    <li><strong>Feature Extraction:</strong> The first step in COLMAP is to detect and extract features from input images using methods like <span style="text-decoration: underline;"><a href="https://en.wikipedia.org/wiki/Scale-invariant_feature_transform">SIFT</a></span> (Scale-Invariant Feature Transform).</li>
    <li><strong>Feature Matching:</strong> COLMAP matches features between pairs of images to find correspondences, which are essential for 3D reconstruction.</li>
    <li><strong>Incremental Reconstruction:</strong> COLMAP incrementally builds the 3D structure by starting with an initial image pair and gradually adding more images, refining the 3D model along the way.</li>
    <li><strong>Bundle Adjustment:</strong> After the incremental reconstruction, COLMAP performs global optimization (<span style="text-decoration: underline;"><a href="https://en.wikipedia.org/wiki/Bundle_adjustment">Bundle Adjustment</a></span>) to refine the camera poses and 3D points.</li>
    <li><strong>Dense Reconstruction:</strong> Finally, COLMAP generates a dense point cloud by performing Multi-View Stereo on the calibrated images.</li>
</ol>
<br/>

<h4>Limitations of COLMAP</h4>
<p>Despite its strengths, COLMAP has some limitations:</p>
<ul>
    <li><strong>Sensitivity to Image Quality:</strong> The accuracy of COLMAP heavily depends on the quality and overlap of the input images. Poorly aligned or low-quality images can lead to errors in the reconstruction.</li>
    <li><strong>Limited Scalability & Robustness:</strong> While COLMAP works well for medium-sized datasets, it can struggle with extremely large or small datasets due to its incremental approach or lack of visual coherence.</li>
    <li><strong>Time-Consuming:</strong> The reconstruction process, especially dense reconstruction, can be time-consuming, making it less suitable for real-time applications.</li>
</ul>

<br/>
<h2 id="sec2">2. Deep Learning-Based Camera Pose Reconstruction</h2>
<p>
    To overcome the limitations of traditional SfM methods like COLMAP, 
    recent developments have introduced deep learning-based approaches for camera pose reconstruction. 
    In the later of this post, we’ll compare two notable methods: <strong>VGGSfM</strong> and <strong>MASt3R</strong>, 
    evaluating their performance and discussing their strengths and weaknesses.
</p>

<h3 id="sec2.1"> VGGSfM: Visual Geometry Grounded Deep Structure from Motion </h3>
<p>
    VGGSfM is a deep-learning-based approach to Structure-from-Motion (SfM) that offers a significant advancement over traditional methods like COLMAP. Unlike conventional SfM, 
    which relies on a sequential, non-differentiable process, VGGSfM employs an end-to-end differentiable pipeline. 
</p>
<figure>
    <img src="./240721_sfm/assets/vgg.jpg" alt="vggsfm overview" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"><strong>Figure 3.</strong> Overview of VGGSfM</figcaption>
</figure>
<p>
    It starts with deep feature extraction and 2D point tracking, 
    followed by simultaneous initialization of all camera poses, differentiable triangulation, and joint optimization through bundle adjustment. 
</p>
    This integration allows VGGSfM to refine each step during training, leading to more accurate and robust 3D reconstructions. 
    The method scales effectively, handles complex scenes, and has shown state-of-the-art results on various benchmarks. 
    In contrast, while COLMAP remains a powerful tool, its reliance on traditional techniques limits its adaptability and performance in comparison to VGGSfM.
</p>

<h3 id="sec2.2"> MASt3R: Grounding Image Matching in 3D with MASt3R </h3>
<p>
    MASt3R (Matching And Stereo 3D Reconstruction) is a cutting-edge framework designed to enhance 3D scene reconstruction and dense image matching, 
    particularly in challenging conditions with extreme viewpoint changes. It builds on the <span style="text-decoration: underline;"><a href="https://github.com/naver/dust3r">DUSt3R</a></span> 
    architecture by employing a shared Vision Transformer (ViT) encoder and cross-attention decoders to capture spatial relationships and 3D geometry between image pairs. 
</p>
<figure>
    <img src="./240721_sfm/assets/mast3r.jpg" alt="mast3r overview" width="100%" onclick="window.open(this.src)">
    <figcaption style="text-align: center; font-size: 15px;">
        <strong>Figure 4.</strong> Overview of MASt3R. <br/> Compared to the DUSt3R
        framework, MASt3R's contributions are highlighted in <span style="color:blue">blue</span>.
    </figcaption>
</figure>
<p>
    MASt3R predicts dense 3D pointmaps and local feature maps, 
    using a coarse-to-fine matching strategy to achieve high accuracy in matching. Its use of a fast nearest-neighbor algorithm and iterative fine-tuning ensures robust performance. 
    Compared to traditional 2D methods like those in COLMAP, MASt3R's 3D-centric approach excels in environments requiring precise visual localization and mapping.
</p>

<h2 id="sec3"> 3. Radiance Fields from Deep-based SfM</h2>
<p>
    To validate geometric feasibility of wild deep-based Structure-from-Motion methodologies, 
    we will show the radiance fields reconsturction from VGGSfM and MASt3R. 
    The objective is to compare their performances and understand the advantages and limitations of each approach.
</p>

<h3 id="sec3.1"> PointCloud Reconstruction</h3>

<table>
    <tr>
        <th>VGGSfM</th>
        <th>MASt3R</th>
    </tr>
    <tr>
        <td><img class="img-fluid" src="./240721_sfm/assets/NLE_vggsfm.jpg" alt="NLE vggsfm"></td>
        <td><img class="img-fluid" src="./240721_sfm/assets/NLE_mast3r.jpg" alt="NLE mast3r"></td>
    </tr>
    <tr>
        <td><img class="img-fluid" src="./240721_sfm/assets/pen_sparse_vggsfm.jpg" alt="pen vggsfm"></td>
        <td><img class="img-fluid" src="./240721_sfm/assets/pen_sparse_mast3r.jpg" alt="pen mast3r"></td>
    </tr>
    <tr>
        <td><img class="img-fluid" src="./240721_sfm/assets/guitar_vggsfm.jpg" alt="guitar vggsfm"></td>
        <td><img class="img-fluid" src="./240721_sfm/assets/guitar_mast3r.jpg" alt="guitar mast3r"></td>
    </tr>
</table>
<br/>

<h3 id="sec3.2"> Radiance Fields Reconstruction</h3>
<table>
    <tr>
        <th>VGGSfM</th>
        <th>MASt3R</th>
    </tr>
    <tr>
        <td><img class="img-fluid" src="./240721_sfm/assets/NLE_vggsfm_2dgs.gif" alt="NLE vggsfm"></td>
        <td><img class="img-fluid" src="./240721_sfm/assets/NLE_mast3r_2dgs.gif" alt="NLE mast3r"></td>
    </tr>
    <tr>
        <td><img class="img-fluid" src="./240721_sfm/assets/pen_sparse_vggsfm_2dgs.gif" alt="pen vggsfm"></td>
        <td><img class="img-fluid" src="./240721_sfm/assets/pen_sparse_mast3r_2dgs.gif" alt="pen mast3r"></td>
    </tr>
    <tr>
        <td><img class="img-fluid" src="./240721_sfm/assets/guitar_vggsfm_2dgs.gif" alt="guitar vggsfm"></td>
        <td><img class="img-fluid" src="./240721_sfm/assets/guitar_mast3r_2dgs.gif" alt="guitar mast3r"></td>
    </tr>
</table>
<br/>

<h4> Summary </h4>
<p>
    MASt3R is not suitable for inverse rendering but provides denser and more diverse point cloud reconstructions compared to VGGSfM.
    VGGSfM's accurate camera pose reconstruction, utilizing Bundle Adjustment, makes it more suitable for inverse rendering. 
    Specifically, VGGSfM's camera pose has less than 0.01 angular distance error compared to COLMAP, while MASt3R's pose has over 0.1 angular distance error.
</p>
<ul>
    <li>
        Both methods are more robust than COLMAP. In my experiment, COLMAP fails to reconstruct all the above datasets.
    </li>
    <br/>
    <li>
        Both methods have shortcomings with limited VRAM capacity, but VGGSfM handles it better.
        VGGSfM can reconstruct over 90 images on a single RTX 4090, whereas MASt3R struggles with over 30 images.
    </li>
</ul>
<br/>

<h4> Further Camera Pose Refinement </h4>
<div class="video-container">
    <video controls style="width: 100%;">
        <source src="./240721_sfm/assets/further_pose_opt.mp4" type="video/mp4">
    </video>
</div>
<p>
    As discussed in <span style="text-decoration: underline;"><a href="https://instantsplat.github.io/">InstantSplat</a></span>, MASt3R (and VGGSfM) poses 
    can serve as a good initial point of the camera pose optimization during radiance fields training (BARF-likes method). 
    Below is the toy experiment of MASt3R + further camera pose optimization (Using Splatfacto):
</p>

<h2 id="closing">Closing</h2>
<p>
    This post compared two 3D point cloud reconstruction methods, VGGSfM and MASt3R. VGGSfM excels in camera pose reconstruction, 
    making it more suitable for inverse rendering, while MASt3R provides denser point clouds but is less appropriate for inverse rendering tasks. 
    Both methods offer robustness over COLMAP and can yield better results in certain scenes, although they require further parameter tuning and optimization.
</p>
<p>
    GitHub Repository: <span style="text-decoration: underline;"><a href="https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r">[Link]</a></span>
</p>
<hr/>
<p>
    You may also like, 
</p>
<br/>