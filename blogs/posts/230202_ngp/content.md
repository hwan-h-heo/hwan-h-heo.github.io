title: Instant NGP Review & Re-implementation
date: Febrary 02, 2023
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
        <li><a href="#sec2"> Background </a></li>
        <li>
            <a href="#sec3"> Method </a>
        </li>
        <ul>
            <li><a href="#sec3.1"> Multi-Level Decomposition </a></li>
            <li><a href="#sec3.2"> Hash Grids Encoding </a></li>
            <li><a href="#sec3.3"> Multi-Resolution Hash Encoding </a></li>
        </ul>
        <li><a href="#conclusion"> Closing </a></li>
    </ul>
</nav>

<br/>
<h2 id="tl-dr">TL; DR</h2>
<p>
    Let's delve into the Instant Neural Graphics Primitive with a Multi-Resolution Hash Encoding, and re-implement this with PyTorch!
</p>
<figure>
    <img src="./230202_ngp/assets/ngp_nerf.gif" alt="Gaussian RT" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"> <strong>Figure 1.</strong> NGP-NeRF </figcaption>
</figure>

<h2 id="intro"> 1.Introduction</h2>
<p> Neural Radiance Fields (NeRF) are a powerful method for 3D scene reconstruction, but they come with significant drawbacks, primarily in terms of slow training and rendering speeds. 
    To address these issues, various studies have explored voxel-based approaches. While these methods can reduce computation time, they often suffer from limited speed improvements or performance trade-offs. </p>
<p>Instant Neural Graphics Primitives (Instant-NGP) offers a breakthrough by utilizing multi-resolution decomposition and hashing, achieving state-of-the-art performance with remarkable speed. </p>
<p>In this article, I review the core of Instant-NGP and provide a PyTorch implementation of its core components.</p>
<ul>
    <li>
        <p>
            project: <span style="text-decoration: underline;"><a href="https://nvlabs.github.io/instant-ngp/">link</a></span>
        </p>
    </li>
</ul>

<h2 id="sec2"> 2. Background</h2><br/>
<h3 id="sec2.1">Positional Encoding</h3>
<p>For high-fidelity scene reconstruction, NeRF typically uses sinusoidal positional encoding:
$$
\gamma(p) = \big (\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \big)
$$</p>
<p>There are alternative encodings, such as Integrated Positional Encoding (IPE) in <a href="https://velog.io/@gjghks950/Mip-NeRF-A-Multiscale-Representation-for-Anti-Aliasing-Neural-Radiance-Fields-Paper-Review">Mip-NeRF</a>, but the fundamental principle remains the same: the information is encoded according to different frequencies.</p>
<p>However, NeRF requires the inference of MLPs—typically 8 layers with 256 or 512 hidden dimensions—for every point in the rendering process. This is one of the primary reasons for NeRF&#39;s slow speed.</p>

<h3 id="sec2.2"> Voxel-based Methods</h3>
<p>One of the main approaches to address these drawbacks is to reduce the computational burden of inference and training by pre-computing and storing data at a few key locations.</p>
<p>This involves:</p>
<ol>
<li>Learning a parametric encoding for the vertices of a 3D voxel grid by introducing learnable parameters, rather than using a fixed positional encoding.</li>
<li>Using linear interpolation to approximate points between vertices, thereby improving speed (as shown in <span style="text-decoration: underline;"><a href="https://alexyu.net/plenoxels/">Plenoxels (CVPR 2022)</a></span>).</li>
</ol>
<figure>
    <img src="./230202_ngp/assets/plenoxel.png" alt="Gaussian RT" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"> <strong>Figure 2.</strong> Voxel-Based NeRF, source: Plenoxels </figcaption>
</figure>
<p>However, voxel-based methods have the disadvantage of requiring significantly more memory compared to NeRF, and they often involve complex training processes, including various regularization techniques.</p>

<h2 id="sec3"> 3. Method </h2><br/>
<h3 if="overview"> Overview </h3>
<p>Instant-NGP uses a similar approach to existing voxel-based methods by mapping parametric encodings to the vertices of a voxel. However, it introduces several key differences:</p>
<ol>
<li><strong>Multi-level Decomposition</strong><br/> The scene is divided into multiple levels, with each level storing information that focuses on different parts of the scene geometry.</li><br/>
<li><strong>Hash Function</strong><br/> As the resolution of voxels increases, the number of points that need to be stored grows cubically. Instead of storing all points on a one-to-one basis, a hash function is used to reduce the memory required.</li>
</ol>
<p>The following figure visualizes the forward process of Multi-Resolution Hash Encoding:</p>
<figure>
    <img src="./230202_ngp/assets/ngp.png" alt="Gaussian RT" width="100%">
    <figcaption style="text-align: center; font-size: 15px;"> <strong>Figure 3.</strong> Multi-Resolution Hash Encoding </figcaption>
</figure>
<ul>
<li>Each voxel&#39;s vertex at different resolutions (<span style="color:#F9B6B9"><strong>red</strong></span> and <span style="color:#B1DBEB"><strong>blue</strong></span>) is stored in a table with a learnable feature vector of dimension $F$. 
    The table-to-vertex mapping is defined through a hash function over the vertex coordinates.
</li><br/>
<li>For any point in space, its encoding is determined by a linear interpolation between the features of all corner vertices of the hypercube to which the point belongs.</li><br/>
<li>This interpolated value is combined with the view-direction encoding and used as input to the decoding network $m(\mathbf{y}; \phi)$.</li>
</ul>
<p>Instant-NGP maximizes the capabilities of parametric encoding and multi-level decomposition, allowing for an extremely shallow decoding network—typically a 2-layer network with 64 hidden dimensions. This leads to much faster point-wise inference and convergence compared to other NeRF models while still achieving SOTA performance.</p>
<p>The next step involves volume rendering using ray casting, similar to other NeRF-like models.</p>
                        
<h3 id="sec3.1"> 3.1. Multi-Level Decomposition</h3>
<p>For a total of $L$ levels, the resolution $N_{l}$ of a voxel at level $l$ is determined as a value between $[N_{\text{min}}, N_{\text{max}}]$, defined as follows:</p>
<div class="math-container">
    $$
            N_{l} 
                := 
            \lfloor N_{\text{min}} \cdot b^{l-1} \rfloor, \\ {} \\
            \text{where } b : = 
            \exp 
            \left( 
                \frac{\ln N_{\text{max}} - \ln N_{\text{min}}}{L-1}  
            \right) .
    $$
</div>

<p>
    To optimize memory usage, rather than declaring a feature table that directly corresponds to each voxel resolution $N_l$, a fixed-size feature table of size $T$ is declared. 
    If the grid size is smaller than $T$, a feature table matching the voxel size is declared to maintain a one-to-one correspondence.
</p>
<p>
    In the following PyTorch custom implementation, the per-level scale $b$ is calculated using the formula above, 
    and the feature tables are initialized accordingly based on whether the voxel size is smaller than $T$ or not.
</p>

```python
self.one2one = []
self.units = []
for i in range(self.n_levels):
    grid_size = (self.units[i]+1) ** 3
    hash_size = 2 ** self.log2_hashmap_size # T in Eqn
    self.one2one.append(grid_size < hash_size)
    self.units.append(int(np.round(self.N_min * (self.per_level_scale ** i))))

    table_size = (self.one2one[i]) * grid_size + (not self.one2one[i]) * hash_size
    torch_hash = nn.Embedding(int(table_size), self.feat_dim) # self.feat_dim : F in Eqn
    nn.init.uniform_(torch_hash.weight, -self.init_std, self.init_std)

    setattr(self, f'torch_hash_{i}', torch_hash) 
```

<ul>
    <li>The <code>self.one2one</code> array indicates which levels have a one-to-one correspondence.</li>
    <li><code>self.units</code> stores the voxel size per level.</li>
</ul>
<br/>

<h3 id="sec3.2"> 3.2. Hash Grids Encoding </h3>
<p>For encoding a point $\mathbf{x} \in \mathbb{R}^{d}$ at each level $l$, the point is first mapped onto a hypercube of size 1 at each level:</p>
<div class="math-container">
    $$ \mathbf{x}_{l} : = \mathbf{x}_{l} \cdot N_{l}
    $$
</div>
<p>This places the point within a hypercube defined by its diagonal vertices $\lfloor \mathbf{x}_{l} \rfloor$ and $\lceil \mathbf{x}_{l} \rceil$.</p>
<p>Subsequently, this hypercube is mapped to the feature table using a hash function:</p>
<div class="math-container">
    $$ 
        h(x)= 
        \left ( 
            \bigoplus_{i=1}^{d} x_{i} \pi_{i}  
        \right ) 
        \quad \text{mod } T
    $$
</div>
<p>where $\pi_i$ are large prime numbers (<em>e.g.,</em> $[1, 2 654 435 761, 805 459 861]$).</p>
<p>After the feature mapping for all $2^d$ vertices is completed, the relative positions within the hypercube are used to interpolate each vertex feature, resulting in the final encoding for level $l$.</p>

<h4 id="hash-grids-trilinear-interpolation"> Hash Grids & Tri-linear Interpolation </h4>
<p>Assume the forward process of Instant-NGP receives $N$ points as input. For a typical NeRF dataset, these points are 3D, so the input shape will be $[N,\ 3]$.</p>
<p>Our goal is to compute:</p>
<ul>
<li>The $2^d$ level-wise corner vertex coordinates of the points $\mathbf{x}$ (<em>i.e.,</em> total $l \times 2^d$ vertices).</li>
<li>The level-wise trilinear interpolation weights for these points.</li>
</ul>

<p>Let&#39;s implement this step by step!</p>
<ol>
<li>
    First, for a given level $l$, distribute the points $\mathbf{x}$ over voxels with grid size $N_l$ and calculate corner vertices by adding offsets ($[0,0,0] \sim [1,1,1]$) to $\lfloor \mathbf{x}_{l} \rfloor$.

```python
corners = []
N_level = self.units[l] # N_min to N_max resolution 

for i in range(2 ** x.size(-1)): # for 2^3 corners 
    x_l = torch.floor(N_level * x)
    offsets = [int(x) for x in list('{0:03b}'.format(i))]
    for c in range(x.size(-1)):
        x_l[..., c] = x_l[..., c] + offsets[c] # 3-dim (x,y,z)
        corners.append(x_l)
    corners = torch.stack(corners, dim=-2)
```
</li>
<li>
    Next, compute trilinear weights using the relative position differences between corners and $\mathbf{x}_l$.

```python
# get trilinear weights 
x_ = x.unsqueeze(1) * N_level 
weights = (1 - torch.abs(x_ - corners)).prod(dim=-1, keepdim=True) + self.eps
```
</li>
</ol>
<p>These processes can be wrapped in a following function.</p>

```python
def hash_grids(self, x):
    # input: x [N, 3]
    # output: 
    #   level_wise_corners: [L, N, 8, 3]
    #   level_wise_weights: [N, 8, L, 1]

    corners_all = []
    weights_all = []

    for l in range(self.n_levels):
        # get level-wise grid corners 
        corners = []
        weights = []
        
        N_level = self.units[l] # N_min to N_max resolution 
        for i in range(2 ** x.size(-1)): # 2^3 corners 
            x_l = torch.floor(N_level * x)
            offsets = [int(x) for x in list('{0:03b}'.format(i))]
            for c in range(x.size(-1)):
                x_l[..., c] = x_l[..., c] + offsets[c] # 3-dim (x,y,z)
            corners.append(x_l)
        corners = torch.stack(corners, dim=-2) # [N, 8, 3]

        # get trilinear weights 
        x_ = x.unsqueeze(1) * N_level # [N, 1, 3]
        weights = (1 - torch.abs(x_ - corners)).prod(dim=-1, keepdim=True) + self.eps # [N, 8, 1]

        corners_all.append(corners)
        weights_all.append(weights)

    corners_all = torch.stack(corners_all, dim=0) # [L, N, 8, 3]
    weights_all = torch.stack(weights_all, dim=-2) # [N, 8, L, 1]
    weights_all = weights_all / weights_all.sum(dim=-3, keepdim=True)

    return corners_all, weights_all 
```
<br/>

<h4 id="table-mapping"> Hash Table Mapping</h4>
<p>The method for table mapping varies depending on whether there is a one-to-one correspondence. Using <code>self.one2one</code> declared in <a href="#sec3.1"><strong>3.1</strong></a>, we handle the two cases:</p>
<ol>
    <li>
        <p>For one-to-one correspondence, the index is directly derived from the coordinates.</p>

```python
for l in range(self.n_levels):
	ids = []

    c_ = c[l].view(c[l].size(0) * c[l].size(1), c[l].size(2)) 
    c_ = c_.int()  
        
    if self.one2one[l]: # grid_size << hash_size 
        ids = c_[:, 0] + (self.units[l] * c_[:, 1]) + ((self.units[l] ** 2) * c_[:, 2])
        ids %= (self.units[l] ** 3)
```
</li>
<li>
    <p>Otherwise, the index is calculated using the hash function defined in <a href="sec3.2"><strong>3.2</strong></a>.</p>

```python
# cf. self.primes = [1, 2654435761, 805459861] 
else: 
    ids = (c_[:, 0] * self.primes[0]) ^ (c_[:, 1] * self.primes[1]) ^ (c_[:, 2] * self.primes[2])
    ids %= (2 ** self.log2_hashmap_size)</code></pre>
```
</li>
</ol>
<p>The entire mapping process can also be wrapped into a single function.</p>

```python
def table_mapping(self, c):
	# input: 8 corners [L, N, 8, 3]
	# output: hash index [L, N * 8]
	
    ids_all = []
    with torch.no_grad():
        for l in range(self.n_levels):
            ids = []
            
            c_ = c[l].view(c[l].size(0) * c[l].size(1), c[l].size(2)) 
            c_ = c_.int()  
            
            if self.one2one[l]: # grid_size << hash_size 
                ids = c_[:, 0] + (self.units[l] * c_[:, 1]) + ((self.units[l] ** 2) * c_[:, 2])
                ids %= (self.units[l] ** 3)
            else: 
                ids = (c_[:, 0] * self.primes[0]) ^ (c_[:, 1] * self.primes[1]) ^ (c_[:, 2] * self.primes[2])
                ids %= (2 ** self.log2_hashmap_size)
            ids_all.append(ids) 
    
    return ids_all # [L * [N*8]]
```

<h3 id="sec3.3"> 3.3. Multi-Resolution Hash Encoding</h3>
<p>We index the feature table to get the feature values for each level declared as <code>nn.Embedding</code>, perform trilinear interpolation, and then concatenate them by level to obtain the final encoding.</p>

```python
def hash_enc(self, corners, weights):
	# input:    corners [L, N, 8, 3]
	#           weights [L, N, 8, 1]
	# output: interpolated embeddings [N, L*F]
	
    level_embedd_all = []
    ids_all = self.table_mapping(corners) # [L * [N*8]]
    
    for l in range(self.n_levels):
        level_embedd = []
        hash_table = (getattr(self, f'torch_hash_{l}'))
        hash_table.to(corners.device)
    
        level_embedd = hash_table(ids_all[l]) # [N*8, 1] -> [N*8, F] 
        
        level_embedd = level_embedd.view(corners.size(1), corners.size(2), self.feat_dim) # [N, 8, F]
        level_embedd_all.append(level_embedd)
    
    # Trilinear Interpolation 
    # weights: [N, 8, L, 1]
    level_embedd_all = torch.stack(level_embedd_all, dim = -2) # [N, 8, L, F]
    level_embedd_all = torch.sum(weights * level_embedd_all, dim=-3) # [N, L, F]
    
    return level_embedd_all.reshape(weights.size(0), self.n_levels * self.feat_dim) 
```
<p>For input $\mathbf{x}$ of shape $[N,\ 3]$, we obtain the multi-resolution hash encoding result.</p>

```python
corners_all, weights_all = self.hash_grids(x)
encodings = self.hash_enc(corners_all, weights_all)
```

<br/>
<h2 id="conclusion"> Closing </h2>
<p>The implementation above demonstrates that by matching the input dimension size, the code can be compatible with any NeRF-like model decoding network.</p>
<p>This flexibility allows us to combine other NeRF models with Multi-Resolution Hash Encoding using this code easily.</p>
<p>However, the implementation may not be as fast as the original Instant-NGP due to several reasons:</p>
<ol>
<li>The PyTorch implementation, unlike the original CUDA/C++ version, incurs additional execution time.</li>
<li>Instant-NGP utilizes the <a href="https://github.com/NVlabs/tiny-cuda-nn">tcnn library</a> for the decoding network, further optimizing inference speed.</li>
<li>There are additional implementation details, such as pruning hypercubes without opaque particles to improve inference efficiency.</li>
</ol>
<hr/>
<p>
    You may also like, 
</p>