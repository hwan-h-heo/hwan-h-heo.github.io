title: Linear Algebra for Deep Learning (2): Change of Basis
date: September 09, 2021
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


<!-- ***- DL Book 이외의 중요한 내용을 일부 추가하였음.***

https://www.deeplearningbook.org/contents/linear_algebra.html -->

---

...continued from [the previous post](./?id=210909_lin_alg_1).

#### TL;DR

This article focuses on the geometric interpretation of linear transformations through the change of basis, exploring various matrix decompositions and their meanings.
Through this, we will understand that the essence of linear transformation lies in its interaction with the coordinate system (basis), and a good understanding of this naturally connects to various decomposition methods and their applications in deep learning.

---

## 3. Change of Basis

### 3.1. Quadratic Form

*   Through the lens of linear systems, a geometric interpretation of the quadratic form, expressed as $A^{-1}MA$, becomes possible. Consider the following:

    <p>
    $$
    \begin{bmatrix} a & c \\ b & d \\ \end{bmatrix} \begin{bmatrix} x \\ y \\ \end{bmatrix} = \begin{bmatrix} x' \\ y' \\ \end{bmatrix}
    $$
    </p>

    If we interpret the above linear system from the perspective of a changing basis, it means that a point $(x,y)$ in a span with basis vectors $(a,b)$ and $(c,d)$ is represented as $(x', y')$ in a span with standard basis vectors $(1,0)$ and $(0,1)$. In other words, the point $(x', y')$ is the interpretation of the point $(x,y)$ from the A-basis in the standard coordinate system.

    Using this, if we interpret $A^{-1}MA$, it can be seen as a transformation from the span defined by $A$ to the standard x,y,z coordinate system, followed by a linear transformation $M$, and then transformed back to the span defined by $A$.
    Therefore, the quadratic form $A^{-1}MA$ can be interpreted as the transformation $M$ as viewed from the perspective of basis A.

### 3.2. Eigen-Decomposition

Let's examine eigen-decomposition from the perspective of a change of basis.

$$
A \vec v = \lambda \vec v \ (= \lambda {\rm I} \vec v ) \\ \rightarrow (A- \lambda \rm I) {\it \vec v} = 0
$$

*   The constant $\lambda$ that satisfies the above equation is called the **eigenvalue**, and the vector $\vec v$ is the **eigenvector**. In other words, an eigenvector is a vector that, when transformed by its own matrix, only changes in scale, and the eigenvalue is the factor by which it scales.

This can be thought of as the axis of rotation of the span transformation (I believe the name 'eigen' or 'own' was given because its direction does not change regardless of the span's transformation).
For the equation to hold when $\vec v$ is not the zero vector, the span must be reduced in dimension. Therefore, the determinant of the transformation $A-\lambda {\rm I}$ in the equation must be 0. From this, we derive the characteristic equation that we memorize:

$$
det(A-\lambda {\rm I} ) = 0
$$

*   Now, let's consider a matrix $V$ whose columns are all eigenvectors of matrix A (a situation where the basis of the span is the eigenvectors, usually called an **eigenbasis**). We can construct a quadratic form $V^{-1}AV$ using this matrix $V$, which results in a diagonal matrix with the eigenvalues as its diagonal components. To write it again:

    <p>
    $$
    V^{-1}AV = \begin{bmatrix} \lambda_1 & &... & & 0\\ . & \lambda_2 & & &. \\ . & & \lambda_3 & &. \\ . & & & \lambda_4 &. \\ 0 & & ... & &\lambda_5 \\ \end{bmatrix}
    $$
    </p>

    From a change of basis perspective, this is the transformation $A$ in the span of $V$. But since $V$ is a transformation whose basis vectors are eigenvectors, it has been transformed into a coordinate system where $A$ only performs scaling. Thus, it must be a diagonal matrix, and because the scaling factors are represented by the eigenvalues, the quadratic form results in this diagonal matrix of eigenvalues.

*   By reconfiguring $A$ from this, we can derive the eigen-decomposition we are familiar with:

    $$
    A = V \text{diag}(\lambda) V^{-1}
    $$

    In essence, eigen-decomposition is a tool that finds the axes of rotation that are independent of the span $A$ through linear transformation, allowing us to expand the space in the desired directions. A simple example of its use is in the iterative application of a linear transformation. Besides the numerical analysis approach—that the forms at the front and back of the quadratic form are inverses of each other—the same interpretation is possible because the very meaning of eigen-decomposition is scaling independent of the span.

    This is also consistent with how in PCA, feature extraction that maximizes the covariance matrix involves selecting the eigenvectors with large eigenvalues from the covariance matrix. Since the way a point is described does not change after a linear transformation through the eigenbasis, PCA can also be interpreted as selecting the basis vectors in order of their contribution.

*   When represented by the eigen-decomposition $A = V \text{diag}(\lambda) V^{-1}$, it is easy to see from the properties of the determinant that $det(A) = \Pi_i \lambda_i$. Therefore, if a matrix is non-singular, it means that $\forall_i \lambda_i \neq 0$. This leads to the definition of positive-definite, semi-definite matrices, etc., which collectively refer to matrices that have eigenvectors that do not reduce the span and do not change the order of the basis vectors.

### 3.3. Similar Matrix

$$
B = P^{-1}AP
$$

*   For a non-singular matrix P, when A and B can be represented as above, A and B are defined as being **similar**. From a change of basis perspective, this means they are the same transformation but expressed in different spans. Also, from the properties of the determinant, we can see that $det(B) = det(A)$. Furthermore, we can deduce that similar matrices have the same eigenvalues through the following:

    <p>
    $$
    B−λI=P^{−1}AP−λP^{−1}P=P^{−1}(A−λI)P \\ \therefore det(B−λI)=det(P^{−1}(A−λI)P)=det(A−λI)
    $$
    </p>

    Using this, we can derive a quite useful property. Let's look at the following:

    *"If there exists an orthogonal matrix P such that $B=P^TAP$ and A is symmetric, then the eigenvectors of A are mutually orthogonal."*

    <p>
    $$
    Av_j = \lambda_j v_j \\ \rightarrow v_i^T (Av_j) = (v_i^TA^T) v_j = (Av_i)^T v_j \\ = \lambda_i v_i^T v_j \\ = \lambda_j v_i^T v_j
    $$
    </p>

    Since $\lambda_i \neq \lambda_j$ (for distinct eigenvalues), we can see that the dot product of any two distinct eigenvectors is 0. If we perform matrix factorization with an orthogonal matrix, we can discover a peculiar property in addition to eigen-decomposition: since the basis vectors are mutually orthogonal, they can be interpreted as a set of perpendicular axes, and thus this orthogonal column matrix itself can be interpreted as a rotation transformation (since the original basis vectors also form an orthogonal relationship).

### 3.4. Singular Value Decomposition

Instead of simply extending eigen-decomposition to non-square matrices, let's reconsider SVD through the concept of orthogonal similarity mentioned above.

*   Let's consider a matrix $A \in \mathbb{R}^{m\times n}$ (where $m \neq n$). This implies a transformation where the span either collapses or expands, and therefore, eigen-decomposition is not defined. Instead, we can construct the following two symmetric matrices:
    1.  $A^TA$: an $n \times n$ symmetric matrix. Its eigenbasis is mutually orthogonal & independent, and its basis forms a span with the rank of the column dimension of $A$.

    2.  $AA^T$: an $m \times m$ symmetric matrix. The rank of its span is the row dimension of $A$.

Let's construct the SVD according to its definition while keeping in mind the meaning of each symmetric matrix.

<p>
$$
A = U \Sigma V^T \\ U, V : \text{each orthonormal}
$$
</p>

Accordingly, if we represent $A^TA$ and $AA^T$:

<p>
$$
A^T A = V \Sigma^T \Sigma V^T \\ A A^T = U \Sigma \Sigma^T U^T
$$
</p>

Here, $V$ is the matrix of orthogonal eigenvectors of $A^TA$, and $U$ is the matrix of orthogonal eigenvectors of $AA^T$. Let's examine the meaning of each part more closely using the first equation. Let's multiply both sides by the matrix $V$ on the right.

<p>
$$
A^TA v_j = \sigma_j^2 v_j \quad ( \text{from } A^TA V = V \Sigma^T \Sigma )
$$
</p>

1.  $AA^T (Av_j) = \sigma_j^2 (Av_j)$:
    By the definition of an eigenvector, $Av_j$ is an eigenvector of $AA^T$. (Here, $v_j$ is one of the columns of $V$, an eigenvector of $A^TA$).

2.  $v_j^T A^TAv_j = \sigma_j^2 v_j^T v_j \quad \rightarrow \quad (Av_j)^T (Av_j) = \sigma_j^2$
    This means $Av_j / \sigma_j$ is a unit eigenvector of $AA^T$, which we call $u_j$. ($u_j \in U$). Therefore,

<p>
$$
Av_j = \sigma_j u_j \\ \rightarrow AV = U \Sigma \\ \rightarrow A = U \Sigma V^T
$$
</p>

Through this, we can see that SVD is defined through the symmetric square matrix $A^TA$ derived from a non-square matrix. Geometrically, this has a meaning similar to eigen-decomposition: since $U$ and $V$ are both square matrices composed of mutually orthonormal columns, they represent rotation transformations. $\Sigma$ represents a scaling transformation with the square roots of the eigenvalues of $A^TA$ (the singular values) as its diagonal elements. Although $U$ and $V$ **may have different dimensions**, they both share the essential property of being rotation transformations.

### 3.5. Moore-Penrose Pseudo-inverse Matrix

Let's look at the derivation process in detail.

In the linear equation:
$$
Ax = b
$$

If $A$ is a non-square or non-singular matrix, we cannot find a general solution using an inverse matrix. Therefore, just as we did with SVD, we can derive it by creating a symmetric and square matrix from $A$ that is easier to handle.

<p>
$$
A^TA x = A^T b \\ \rightarrow x = (A^T A)^{-1} A^T b
$$
</p>

<p>
$$
(A^TA)^{-1} A^T = \{ (V \Sigma^T U^T) (U \Sigma V^T) \}^{-1} V \Sigma^T U^T \\ = (V \Sigma^T \Sigma V^T)^{-1} V \Sigma^T U^T \\ = V (\Sigma^T \Sigma)^{-1} V^T V \Sigma^T U^T \\ = V (\Sigma^T \Sigma)^{-1} \Sigma^T U^T
$$
</p>

*(Note: The result $V \text{diag}(\sigma^{-1}) U^T$ is for the pseudo-inverse of A, not the full expression $(A^TA)^{-1}A^T$. The pseudo-inverse $A^+$ is $V \Sigma^+ U^T$, where $\Sigma^+$ is $\Sigma$ with its non-zero elements inverted.)*

The pseudo-inverse matrix is a matrix whose singular values are the reciprocals of the singular values of A.
The expression at the beginning actually originates from the problem of considering L2 regularization in a least squares solution context.

<p>
$$
\frac{d}{dx} \left( \| Ax - b\|_2^2 + \alpha \| x\|_2^2 \right) = 0 \\ \rightarrow (A^TA + \alpha I)x = A^Tb \\ \rightarrow \hat{x} = (A^TA + \alpha I)^{-1} A^T b
$$
</p>

There is a geometric approach to explain why the described least squares solution has the minimum $\|x\|_2$ norm, but it is difficult to illustrate with diagrams, so I will omit it. The concept is not too difficult, on the level of high school geometry and vectors (though it's no longer high school level...).

## 4. Trace Operator & Norm

### 4.1. Norm

<p>
$$
\|x \|_p = \left(\sum_i |x_i|^p\right)^{\frac{1}{p}}
$$
</p>

<p>
$$
\|A\|_F = \sqrt{\sum_{i,j} A_{i,j}^2}
$$
</p>

### 4.2. Trace Operator

$$
\text{tr}(A) = \sum_i A_{i,i}
$$

The trace operator has several interesting properties. I will describe some useful properties with simple proofs.

1.  **Linearity**: $\text{tr}(cA + dB) = c \cdot \text{tr}(A) + d \cdot \text{tr}(B)$

2.  $\text{tr}(A) = \text{tr}(A^T) \quad$ (by definition)

3.  **Cyclic Property**: $\text{tr}(AB) = \text{tr}(BA)$

    <p>
    $$
    \sum_i (AB)_{i,i} = \sum_i \sum_j a_{i,j} b_{j,i} = \sum_j \sum_i b_{j,i}a_{i,j} = \sum_j (BA)_{j,j}
    $$
    </p>

4.  $\text{tr}(A) = \sum \lambda$ (sum of eigenvalues)

    <p>
    $$
    \text{tr}(A) = \text{tr}(V \text{diag}(\lambda)V^{-1} ) \\ = \text{tr}(V^{-1} V \text{diag}(\lambda)) \\ = \text{tr}(\text{diag}(\lambda)) \\ = \sum \lambda
    $$
    </p>

5.  **Frobenius Norm**: $\|A\|_F^2 = \text{tr}(AA^T) = \text{tr}(A^TA)$
    Let's use SVD, $A = U\Sigma V^T$.
    <p>
    $$
    \|A\|_F^2 = \text{tr}(A^TA) = \text{tr}((U\Sigma V^T)^T (U\Sigma V^T)) \\ = \text{tr}(V\Sigma^T U^T U\Sigma V^T) = \text{tr}(V\Sigma^T \Sigma V^T) \\ = \text{tr}(\Sigma^T \Sigma V^T V) = \text{tr}(\Sigma^T \Sigma) = \sum \sigma_i^2
    $$
    </p>
    Therefore,
    <p>
    $$
    \|A\|_F = \sqrt{\text{tr}(A^TA)}
    $$
    </p>

## 5. Principal Component Analysis

The derivation of the PCA formula generally uses Lagrangian multipliers, so I will explain it again later after discussing duality.
Instead, I will briefly describe the commonly used covariance matrix and explore the relationship between PCA and eigen-decomposition using a method other than Lagrangian multipliers.

### 5.1. Covariance Matrix

$$
\text{Cov}(A) = \frac{1}{n-1} (A - \vec{\mu})^T (A - \vec{\mu})
$$
*(Note: The formula often uses $1/n$ or $1/(n-1)$. Let's assume the data is centered.)*

In the above equation, let $X = A - \vec{\mu}$. Then, eigen-decomposition is possible.

<p>
$$
X^TX = V \Lambda V^{-1} \\ = V \Lambda V^T \quad (\because X^TX \text{ is symmetric})
$$
</p>

*From this, we know the matrix of eigenvectors V is orthogonal, so $V^{-1}=V^T$.*
Therefore, the eigenbasis of the covariance matrix is mutually orthogonal. Since the dot product between the basis vectors that make up the covariance matrix is 0, they are also said to be uncorrelated.

The reason PCA proceeds with feature extraction in the direction that maximizes this is that the covariance matrix constructed from the data represents how spread out the data is. That is, by proceeding in a direction that preserves the maximum variance, the loss of information is minimized even when dimensionality reduction occurs (lossy compression).

### 5.2. Principal Component Analysis:

We can examine the relationship between PCA and eigen-decomposition through the final derived form of the encoder matrix in the following equation. According to the derivation in the Deep Learning book, PCA is equivalent to:

<p>
$$
\arg\max_d \text{Tr}(d^T X^TXd) \\ \text{where } d^T d = 1
$$
</p>

It can be easily seen that $d^TX^TXd$ defines a quadratic form.
To add a brief comment on the formula, the constraint $\|d\|=1$ is imposed because when solving the maximization problem, the objective function value would simply increase by increasing the magnitude of the vector d.

Since $X^TX$ is a symmetric matrix, we can find an eigen-decomposition $X^TX = V\Lambda V^T$. Let $d = Vd'$, then:

<p>
$$
d^TX^TXd = (Vd')^T (V \Lambda V^T) (Vd') \\ = d'^T (V^T V \Lambda V^T V) d' \\ = d'^T \Lambda d'
$$
</p>

This equation represents an ellipsoid in the span formed by the eigenbasis, and it means we are choosing the largest axis among those that form this ellipse. To be more specific, let's derive the ellipse equation:

<p>
$$
\begin{bmatrix} a_1 & ... & a_n \end{bmatrix} \begin{bmatrix} \lambda_1 & & 0\\ \vdots & \ddots & \vdots \\ 0 & & \lambda_n \\ \end{bmatrix} \begin{bmatrix} a_1 \\ \vdots \\ a_n \\ \end{bmatrix} \\ = \lambda_1 (a_1)^2 + \lambda_2 (a_2)^2 + ... + \lambda_n (a_n)^2 \\ \text{where } d'^T = \begin{bmatrix} a_1 & ... & a_n \end{bmatrix}
$$
</p>

Thus, we can confirm that it is an ellipse in the coordinate system formed by the eigenbasis.

Besides the geometric explanation, we can also briefly show mathematically why the maximum value occurs when $d$ is an eigenvector. Let's see. If d is an eigenvector of $X^TX$:

<p>
$$
d^T X^TX d \quad \text{where } d^T d = 1 \\ = d^T (\lambda d) \\ = \lambda (d^Td) \\ = \lambda
$$
<p>

Therefore, when $d$ is an eigenvector, the value of the expression is equal to its corresponding eigenvalue. Thus, we can confirm that PCA is the task of finding the eigenvector corresponding to the maximum eigenvalue. In other words, the task of finding the eigenvector corresponding to the maximum eigenvalue of $\text{Cov}(X)$ is PCA.

This can also be inferred through the intuition that comes from the definition of eigen-decomposition: a large eigenvalue corresponds to an axis that contributes significantly to describing the data (because its sensitivity to scaling is high). Therefore, preserving these axes in order of their magnitude is equivalent to keeping only the axes that contribute the most to forming the overall data.