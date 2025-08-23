title: Generalized Winding Number
date: August 09, 2025
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

---

### TL; DR
 
Let's explore the definition of Winding Number and how its 3D extension, Generalized Winding Number, can be used in SDF calculation! 
<br/>

## Introduction 

Calculating the Signed Distance Field (SDF) in graphics is essential for animation, collision detection, physical simulation, and more. 
As its name suggests, SDF is a 'signed distance function,' which outputs the *closest distance from a point to an object*, with a positive value if the point is outside the object and a negative value if it is inside. 

<figure id="figure-2" >
  <img src='./250809_gwn/assets/image-3.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Signed Distance Field</figcaption>
</figure>

This can be accurately obtained by solving a non-trivial PDE problem called the 'Eikonal Equation,' but since accurately finding the solution to this equation for all space is computationally very difficult, we usually first calculate the easily obtainable 'distance' and then determine the 'inside vs. outside' distinction to assign the sign. 

However, distinguishing 'outside vs. inside' is also a non-trivial problem, much more difficult than calculating the distance. 
Although there are methods to determine inside/outside using the `flood fill` algorithm, they cannot fix holes or self-intersections in the mesh. 

In this post, we will explore the ***Generalized Winding Number***, which defines SDF elegantly and possesses the ability to repair issues like holes or self-intersections in broken meshes. 

## What is Winding Number?
<br/>

<figure id="figure-1" >
  <img src='./250809_gwn/assets/image.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Winding Number</figcaption>
</figure>

Winding Number is an integer value that indicates ***how many times a closed curve is wrapped*** around a specific point. This concept is important in various mathematical fields, such as complex analysis and algebraic topology, as well as in computer graphics. 

In this post, we will look at an intuitive understanding of the winding number as defined by the Cauchy integral, and how it is used for SDF calculation of polygonal meshes.

### 1. Intuition

***Winding Number*** literally means the 'number of wraps.' 

Imagine a point $p$ on a plane and a closed curve $C$ that does not pass through that point. The value that indicates *how many rotations are completed around point $p$ when walking one full circuit along curve $C$* is the Winding Number.

- Rotation in the **counter-clockwise direction** is counted as **positive (+)**,
- Rotation in the **clockwise direction** is counted as **negative (-)**.
- If the curve does not enclose point $p$, the winding number is **0**.

Based on this intuition, let's extend this to more mathematical terms.

### 2. Definition w/ Cauchy Integral

In the complex plane, the winding number $n(\gamma, a)$ for a closed path $\gamma$ and a point $a$ not on the path is defined by the following Cauchy integral:

<p>
$$
n(\gamma, a) = \frac{1}{2\pi i} \oint_\gamma \frac{1}{z-a} dz
$$
</p>

This integral value is always an integer, representing how many times $\gamma$ wraps around point $a$.

---

#### Why can Winding Number be defined by Cauchy integral?

1. **Meaning of $ \frac{1}{z-a} $:** This function has a pole at point $a$ and is also the derivative of the complex logarithm function. That is, differentiating $ \log(z-a) $ gives $ \frac{1}{z-a} $.

2. **Meaning of the Integral = Change in Angle:** The complex logarithm function $ \log(w) $ can be expressed as:
$$
\log(w) = \ln|w| + i \arg(w)
$$
    Here, $\arg(w)$ is the argument of the complex number $w$, meaning the angle from the origin. 

    ![image.png](./250809_gwn/assets/image-2.png)

    Therefore, the expression $$\oint_\gamma \frac{1}{z-a} dz$$ is equivalent to finding the change in $\log(z-a)$ along the path $\gamma$. Since the path is closed, the starting and ending values of the $ \ln|z-a| $ part are the same, so they do not affect the integral value ([Cauchy’s Integral Theorem](https://en.wikipedia.org/wiki/Cauchy%27s_integral_theorem)). 

    Ultimately, this integral represents the total change in the angle, i.e., $\arg(z-a)$, relative to point $a$ along the path $\gamma$.
    
3. **Division by $ 2\pi i $:** When completing one full circuit along the path $\gamma$, the angle $ \arg(z-a) $ changes by an ***integer multiple*** of $ 2\pi $. That is, if $k$ laps are completed, the total angle change will be $ 2\pi k $. Therefore, dividing the integral value by $ 2\pi i $ yields only the integer $k$, which is the total number of rotations.

<aside>
💡 <strong id="strong-2" > Intuitive Summary</strong> : The Cauchy integral formula is essentially a method for calculating the total change in angle around a reference point while moving along a path. $ \frac{1}{z-a} $ measures the angle change, and the integral sums these changes, then dividing by $ 2\pi i $ cleanly extracts the integer value of the number of rotations.
</aside>

---

## Generalized Winding Number

The concept of winding number in a 2D plane extended to 3D space and general surface meshes is the **Generalized Winding Number (GWN)**. 

---

### 1. Discretization of Winding Number

Regarding the definition of winding number examined above, without loss of generality, if we assume the reference point $p$ is the origin, the winding number can be defined by *integrating the angle change in spherical coordinates*.

<p>
$$
w(p) = \frac{1}{2\pi} \int_C d\theta
$$
</p>

Intuitively, this is equivalent to projecting the curve $C$ onto a unit circle around $p$, and dividing the signed length of that projected path by $2\pi$.

If curve $C$ is a polyline composed of several segments, or piecewise linear, this integral can be precisely discretized into the sum of the angles formed by each segment.

<p>
$$
w(p) = \frac{1}{2\pi} \sum_{i=1}^n \theta_i
$$
</p>

Here, $\theta_i$ is the signed angle formed by two consecutive vertices $c_i, c_{i+1}$ on the curve and the point $p$.

### 2. Winding Number in 3D: Solid Angle

Winding Number can be immediately generalized to 3D. In this case, the **angle** in 2D space is replaced by the **solid angle, ($\Omega$)** in 3D. 
The solid angle is a measure of how much space a surface occupies when viewed from a point.

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Angle_solide_coordonnees.svg/500px-Angle_solide_coordonnees.svg.png' width=40%>

The winding number of a closed surface $S$ with respect to point $p$ is defined as the total solid angle subtended by $S$ at $p$, divided by the solid angle of the entire sphere, $4\pi$.

<p>
$$
\omega(p) = \frac{1}{4\pi} \int_{S} \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|^3} dA
$$
</p>

- In this formula, $p$ is a point in 3D space for which the winding number is being calculated, $S$: closed surface (mesh), $x$: a point on the surface $S$, $\mathbf{n}_x$: the normal vector at point $x$. 

- ${(x-p) \cdot \mathbf{n}_x} / {\|x-p\|^3} dA$: This term represents the differential solid angle of the infinitesimal area $dA$ viewed from point $p$. 

    For any area $dA$ on the surface, the area actually 'seen' from the observation point $p$ is not the actual size of $dA$, but the area $dA_{\perp}$ projected onto the tangent space perpendicular to the direction. 

    <p>
    $$
    dA_{\perp} = dA \cdot \cos(\theta)
    $$
    </p>
    
    Now, if we consider projecting this onto a sphere of radius 1 centered at $p$, the solid angle can be expressed as $(r = \||x-p\||)$: 
    
    <p>
    $$
    d\Omega = \frac{dA_{\perp}}{r^2} = \frac{dA \cdot \cos(\theta)}{r^2}
    $$
    </p>
    
    At this point, since $\cos(\theta)$ is given by the dot product:
    
    <p>
    $$
    \cos(\theta) = \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|} \\ \because \|\mathbf{n}_x\| = 1
    $$
    
    </p>
    
    Using this, the formula for the solid angle can be organized as follows: 
    
    <p>
    $$
    \begin{aligned}
    d\Omega &= \frac{dA}{r^2} \cdot \left( \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|} \right) \\
    &= \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|^3} dA
    \end{aligned}
    $$
    </p>

- ${1}/{4\pi}$: This is the normalization term, equal to the solid angle of the entire sphere, $4\pi$. 



Similarly, if the surface is composed of a triangle mesh, it can be precisely discretized into the sum of the solid angles $\Omega_f$ created by each triangle.

<p>
$$
w(p) = \sum_{f=1}^m \frac{1}{4\pi} \Omega_f(p)
$$
</p>

Here, $\Omega(p, T)$ denotes the solid angle subtended by triangle $T$ at point $p$.

<aside>
💡 <strong id="strong-2" > Summary </strong> : Although defined mathematically by the Cauchy integral formula, one can think of it as calculating the ratio of the projected surface area of all mesh faces onto a unit sphere centered at point $p$. Just as the winding number in 2D calculates the counter-clockwise wrap as positive, GWN is calculated based on the principle that the front side of the surface (Normal direction) is seen as positive, and the back side is seen as negative, which are then summed up. That is, on the mesh surface this value will be close to 0.5, inside the mesh it will be close to 1, and outside the mesh it will be close to 0. 
</aside>

---

## GWN Field for SDF

<br/>

### SDF from GWN

The biggest advantage of GWN is **Robustness**. While traditional methods require the mesh to be "clean" (e.g., no holes, no self-intersections, normal-consistent, watertight manifold), GWN can calculate meaningful values even on **Broken Meshes** where these assumptions are violated.

<figure id="figure-2" >
  <img src='./250809_gwn/assets/image-5.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> GWN + UDF from Polygonal Mesh </figcaption>
</figure>

As we saw earlier, $w(p)$ is 0.5 on the mesh surface, so the isosurface of the implicit function $f$ defined as $$ f(p) = \omega(p) - 0.5$$ is the mesh surface defined through GWN. By applying this, even for a *broken mesh* where it is difficult to determine the inside/outside, a *repaired mesh* with a well-defined SDF can be calculated. 


1. **Sign Determination**

    - If $\omega(p) \approx 1$, the point $p$ is **inside**. ($\text{sign}(p) = -1$)
    - If $\omega(p) \approx 0$, the point $p$ is **outside**. ($\text{sign}(p) = +1$)
    - We can use a threshold of 0.5 to determine the sign, e.g., `if ω(p) > 0.5 then inside else outside`. This method works very well even if the mesh is not closed. Thanks to its integral-based definition, GWN does not rely on the topological connectivity or geometrical perfection of the input mesh. 
2. **Distance Calculation**: Separate from the sign, calculate the geometrical shortest distance from point $p$ to the mesh $S$. (This can be efficiently calculated using a BVH.)
3. **SDF Combination**: Combine the sign and distance obtained from the above two steps to get the final SDF value.

<p>
$$
\text{SDF}(p) = \text{sign}(0.5-w(p)) \times \text{distance}(p, S)
$$
</p>

---

### Why GWN?

GWN's greatest strength is its **Robustness**. It calculates meaningful values even if the input mesh is **open, non-manifold, or has other issues**.

- **Harmonic Function**: GWN is a Harmonic Function in all space except the input mesh itself. This means that the function value at any point equals the average value around that point, which minimizes unnecessary oscillation.

    - **Why is this important?**: Thanks to this property, even if the mesh is incomplete, the Winding Number Field around it changes very smoothly and predictably. This is because GWN can be expressed as the sum of individual harmonic functions generated by each facet.
    
- **Jump Discontinuity** at the boundary: GWN has a jump discontinuity where the value changes abruptly by $ \pm 1 $ when crossing the faces of the input mesh.

    - **Intuitive Meaning**: The SDF function approaches 0 as it gets closer to the surface, making the distinction between inside and outside ambiguous (Eikonal Equation). In contrast, GWN maintains a clear difference in values between the inside and outside, no matter how close it approaches the surface.
    
    - **Why is this important?**: This property makes GWN a very reliable "confidence measure" for determining whether a point is inside or outside.


<figure id="figure-2" >
  <img src='./250809_gwn/assets/image-4.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Mesh Repair with GWN</figcaption>
</figure>
    
    
These characteristics make GWN very effective for determining the inside/outside of an incomplete mesh and for repairing the mesh based on this using energy minimization techniques like **Graphcut**. 

--- 

## Conclusion 

Through this post, we explored the definition of winding number, its generalization to Generalized Winding Number, and how this concept can be used for Signed Distance Field calculation. 

Since watertight conversion is essential in mesh pre-processing for 3D generation (cf: [Building Large 3D Generative Model (1)](/blogs/posts/?id=250702_building_large_3d_1)), I think it is necessary for those working in this field to understand in detail the methods used in graphics to construct watertight meshes. 

Of course, GWN also has a high computational cost and is not a perfect way to fix broken meshes, so methods for calculating SDF using diffusion simulation of normal vectors, such as the heat method ([GSD](https://nzfeng.github.io/research/SignedHeatMethod/index.html)), have been proposed. In the next post, we will explore methods for calculating unsigned distance fields through heat diffusion, and GSD. 

- **Reference**: [Robust Inside-Outside Segmentation using Generalized Winding Numbers" (Jacobson et al., 2013)](https://dl.acm.org/doi/10.1145/2461912.2461916)

***Stay Tuned!***