:::{.container .portfolio-details-container .col-11}

:::{.row .gy-4}

:::{.col-lg-8}
:::{.portfolio-description}

## Project Overview

This project is a comparative study focused on evaluating the performance of two different deep learning-based Structure-from-Motion (SfM) methodologies, **VGGSfM** and **Mast3r**, in the context of Radiance Fields reconstruction using Gaussian Splatting.  The primary objective was to understand the strengths and weaknesses of each approach when applied to wild, internet-sourced image datasets.

By implementing and testing both VGGSfM and Mast3r pipelines for generating 3D point clouds and camera poses, and subsequently using these outputs for 2D Gaussian Splatting, this project provides insights into the suitability of each SfM method for inverse rendering tasks. The comparison includes qualitative visual assessments of the reconstructed point clouds and Radiance Fields, as well as a quantitative analysis of their characteristics.

:::{.mt-4}
**Key Contributions:**

- Implemented Gaussian Splatting pipelines using both VGGSfM and Mast3r for 3D reconstruction.
- Developed a script to convert MASt3R's output to a COLMAP-compatible format for easier integration with existing visualization and rendering tools.
- Conducted experiments on diverse scenes to compare the performance of VGGSfM and Mast3r.
- Analyzed and summarized the results, highlighting the advantages and limitations of each method for Radiance Fields reconstruction.

:::
:::
:::

:::{.col-lg-4}
:::{.portfolio-info}

### Project Details

- **Category**: Research, Comparative Study, 3D Reconstruction
- **Project URL**: [Project Github](https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r)
- **Skills Demonstrated**: Structure from Motion (SfM), Gaussian Splatting, 3D Visualization, Comparative Analysis

:::
:::

:::
:::

:::{.container .col-11}
:::{.section-title}
:::{#features .portfolio-description}

## Key Features & Findings

:::{.row .feature-icons}
:::{.row}
:::{.col-md-6 .icon-box data-aos="fade-up"}

##### MASt3R/VGGSfM to COLMAP Conversion

To facilitate the use of MASt3R's output within standard 3D visualization and rendering pipelines, I developed a Python script to convert MASt3R's results into a COLMAP-compatible format. This enables seamless integration with tools that are designed to work with COLMAP's output structure.

:::
:::{.col-md-6 .icon-box data-aos="fade-up" data-aos-delay="100"}

##### Online COLMAP Results Viewer

For easy visualization of the reconstructed point clouds and camera poses from both VGGSfM and Mast3R (after conversion), I created an online viewer based on Viser. This viewer allows for interactive exploration of the 3D scenes directly in a web.

:::
:::
:::

:::{.row}
:::{.col-lg-12 data-aos="fade-up"}

##### Findings

Key observations from the comparative study:

- **MASt3R:** Provides denser and more diverse point cloud reconstructions, but camera pose accuracy is less suitable for inverse rendering.
- **VGGSfM:** Offers more accurate camera pose reconstruction due to Bundle Adjustment, making it better suited for inverse rendering tasks despite sparser point clouds.
- **Robustness:** For sparse views, both VGGSfM and MASt3R are more robust than traditional COLMAP, successfully reconstructing datasets where COLMAP fails.
- **VRAM Efficiency:** VGGSfM demonstrates better VRAM efficiency, capable of handling larger datasets compared to MASt3R.
- **Pose Refinement Potential:** MASt3R and VGGSfM poses can serve as effective initializations for camera pose optimization during Radiance Field training.

:::
:::
:::
:::

:::{.section-title}
:::{.portfolio-description}

## Results Visualization

**1) COLMAP PointCloud Comparison**

<table>
<tr>
<th>MASt3R</th>
<th>VGGSfM</th>
</tr>
<tr>
<td><img src="https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r/raw/main/assets/pen_sparse_mast3r.PNG" alt="Pen Sparse MASt3R" style="max-width: 100%; height: auto;"/></td>
<td><img src="https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r/raw/main/assets/pen_sparse_vggsfm.PNG" alt="Pen Sparse VGGSfM" style="max-width: 100%; height: auto;"/></td>
</tr>
</table>

**2) Radiance Fields Reconstruction Comparison**

<table>
<tr>
<th>MASt3R</th>
<th>VGGSfM</th>
</tr>
<tr>
<td><img src="https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r/raw/main/assets/pen_sparse_mast3r_2dgs.gif" alt="Pen Sparse MASt3R 2DGS" style="max-width: 100%; height: auto;"/></td>
<td><img src="https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r/raw/main/assets/pen_sparse_vggsfm_2dgs.gif" alt="Pen Sparse VGGSfM 2DGS" style="max-width: 100%; height: auto;"/></td>
</tr>
</table>

**3) MASt3R + Camera Pose Optimization**

:::{.text-center}
<video controls muted loop style="max-width: 70%;">
<source src="vid_poseopt.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
:::
:::

:::
:::
