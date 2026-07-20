:::{.container .portfolio-details-container .col-11}

:::{.row .gy-4}

:::{.col-lg-8}
:::{.portfolio-description}

## Project Overview

As a personal project, I developed the **2D Gaussian Splatting Viewer**, a first publicly available tool designed to enhance the accessibility and usability of the "[2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)" research. Recognizing the need for a user-friendly interface to explore and manipulate 2D Gaussian Splatting data, I built this viewer leveraging existing libraries and adding significant features to improve visualization, editing, and overall workflow.

This project addresses the gap in readily available tools for interacting with 2D Gaussian Splatting, making it easier for researchers and enthusiasts to delve into this exciting field. By providing intuitive controls and a range of functionalities, the viewer empowers users to gain deeper insights and experiment with 2D GS scenes effectively.

:::{.mt-4}
**Key Contributions:**

- Developed an intuitive and feature-rich viewer from the ground up, based on the 2D Gaussian Splatting.
- Integrated various rendering modes for comprehensive data visualization.
- Implemented editing and transformation tools to enable scene manipulation.
- Added training capabilities directly within the viewer for streamlined experimentation.
- Continuously improved performance and added new features based on user needs and research advancements.

:::
:::
:::

:::{.col-lg-4}
:::{.portfolio-info}

### Project Details

- **Category**: Web Application, 3D Visualization Tool
- **Project URL**: [Viewer Github](https://github.com/hwanhuh/2D-GS-Viser-Viewer)
- **Skills Demonstrated**: Python, Viser, CUDA, Gaussian Splatting, Rasterization

:::
:::

:::
:::

:::{.container .col-11}
<section id="features" class="features section">

## Key Features

---

:::{.viewer-feature-group}

#### Rendering & Training Enhancements

:::{.viewer-feature-grid .viewer-feature-grid-2}
:::{.viewer-feature-card data-aos="fade-up"}
<video class="feature-img" autoplay muted loop playsinline preload="metadata" aria-label="Rendering mode controls in the 2D Gaussian Splatting viewer">
    <source src="assets/rendering-suite.mp4" type="video/mp4">
</video>

##### Advanced Rendering Suite

To enable comprehensive analysis, the viewer supports a diverse set of rendering types, including RGB, Edge, Normal, Depth, and more. This allows users to visualize 2D GS data from multiple perspectives.

:::
:::{.viewer-feature-card data-aos="fade-up" data-aos-delay="100"}
<video class="feature-img" autoplay muted loop playsinline preload="metadata" aria-label="Interactive 2D Gaussian Splatting training workflow">
    <source src="assets/training-workflow.mp4" type="video/mp4">
</video>

##### Integrated Training Capability

For a more seamless workflow, the viewer supports training functionality directly. Users can train 2D GS models and monitor the learning process in real-time within the same application.

:::
:::
:::

:::{.viewer-feature-group}

#### Editing & Transform Tools for Scene Manipulation

:::{.viewer-feature-grid .viewer-feature-grid-3}
:::{.viewer-feature-card data-aos="fade-up"}
<video class="feature-img" autoplay muted loop playsinline preload="metadata" aria-label="General viewer controls and crop box tools">
    <source src="assets/general-controls.mp4" type="video/mp4">
</video>

##### Intuitive General Controls

The viewer features user-friendly general controls, including Render Type selection, Cropbox for region of interest, and Pointcloud visualization mode.

:::
:::{.viewer-feature-card data-aos="fade-up" data-aos-delay="100"}
<video class="feature-img" autoplay muted loop playsinline preload="metadata" aria-label="Splat editing and mesh export tools">
    <source src="assets/editing-and-export.mp4" type="video/mp4">
</video>

##### Interactive Editing and Mesh Export

Users can directly modify and save splats within the viewer.  Also, the implementation of mesh export functionality enables seamless integration of 2D GS scenes into other 3D applications and workflows.

:::
:::{.viewer-feature-card data-aos="fade-up" data-aos-delay="200"}
<video class="feature-img" autoplay muted loop playsinline preload="metadata" aria-label="Rigid transformation controls for Gaussian Splatting scenes">
    <source src="assets/rigid-transform.mp4" type="video/mp4">
</video>

##### Rigid Transformation Capabilities

To provide spatial control, the viewer supports Rigid Transform functionality, enabling users to freely adjust the position, rotation, and scale of 2D GS Scenes.

:::
:::
:::

:::{.row}
:::{.col-lg-12 data-aos="fade-up"}

#### Detailed Feature List

---

Beyond the highlighted features, the viewer includes:

- **Comprehensive Render Types**: RGB, Edge, Normal, View-Normal, Depth, Depth-to-Normal, Depth-Distortion, Curvature
- **Disk Visualization**: Option to visualize splats as disks for clarity.
- **Edit & Save Splats**: Interactive modification and saving of splat data.
- **Mesh Export (Edit Tab)**: Export edited scenes as standard mesh formats.
- **Render Path and Preview**: Tools for generating and previewing camera paths for video rendering.

:::
:::
</section>

:::
