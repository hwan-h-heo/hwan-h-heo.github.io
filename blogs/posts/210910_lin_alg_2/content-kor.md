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

- [이전 글](./?id=210909_lin_alg_1)에 이어...

#### TL;DR
이 글에서는 basis 변화를 통해 선형변환을 해석하는 기하학적 관점을 중심으로, 다양한 행렬 분해 (decomposition)와 그 의미를 살펴본다.
이를 통해 선형변환의 본질은 좌표계 (basis) 와의 상호작용 속에 있고, 이를 잘 이해하면 다양한 분해 방식과 딥러닝에서의 응용까지 자연스럽게 연결된다는 것을 이해해보자.

---

## 3. Change of basis

### 3.1. **Quadratic Form**

- 선형방정식계를 통해서, $A^{-1}MA$  꼴로 나타내지는 quadratic form 의 기하학적인 해석이 가능하다. 다음을 보자
    
    <p>
    $$
    \begin{bmatrix} 
    a & c \\
    b & d \\
    \end{bmatrix} 
    \begin{bmatrix} 
    x  \\
    y  \\
    \end{bmatrix} 
     = 
    \begin{bmatrix} 
    x'  \\
    y'  \\
    \end{bmatrix} 
    $$
    </p>
    
    위 선형방정식 계를, basis 가 변하는 관점에서 해석한다면 basis 가 $(a,b), \ (c,d)$인 span 에서 $(x,y)$인 점이, basis 가 $(0,1),\ (1,0)$인 span 에서 $(x', y')$ 으로 나타난다는 의미이다. 즉 점$(x', y')$은, $A$에서의 $(x,y)$점을 일반적인 좌표계에서 해석한 것이다. 
    
    이를 이용하여 $A^{-1}MA$  를 해석해보자면, $A$로 정의되는 span 에서 일반적으로 사용하는 x,y,z 좌표계로의 변환 이후, $M$ 이라는 선형변환이 이루어지고 이를 다시 $A$가 정의하는 span으로 바꾼 것이라고 해석할 수 있다. 
    즉 quadratic form $A^{-1}MA$ 는 $A$에서의 변환 $M$ 이라고 해석 가능하다. 
    

### 3.2. **Eigen-Decomposition**

basis 의 변화라는 관점에서 eigendecomposition 을 살펴보자. 

<p>
$$
A \vec v = \lambda \vec v \ 
(= \lambda {\rm I}  \vec v ) \\ \rightarrow (A- \lambda \rm I) {\it \vec v} = 0
$$
</p>

- 위 식을 만족시키는 상수 $\lambda$ 의 값을 eigenvalue, 벡터 $\vec v$를 eigenvector 라고 한다. 즉 자기 자신을 선형변환 했을 때, 오직 scale 만 바뀌는 벡터가 eigenvector, 바뀌는 scale 정도가 eigenvalue 로 정의된다.

이는 곧 span 변화의 회전축이라고 생각할 수 있다 (span의 변화에 상관없이 그 위치가 변하지 않기 때문에 'eigen' 이라는 이름이 붙지 않았나 생각한다). 
$\vec v$ 가 영벡터가 아닐 때 위 식을 만족하기 위해서는 span이 축소되는 경우밖에 없으며, 따라서 위 식에서 변환 $A-\lambda {\rm I}$ 의 determinant 값은 0이 되어야만 한다. 이로부터 우리가 암기하고 있는 

$$
det(A-\lambda {\rm I} ) = 0
$$

라는 characteristic equation 이 유도된다. 

- 이제 행렬의 모든 column이 모든 eigenvector로 이루어져있는 행렬 $V$를 생각하자 (span의 basis가 eigenvector 인 상황, 보통 eigenbasis 라고 부른다). 위 행렬 $V$를 통해서 $V^{-1}AV$ 라는 quadratic form을 구성할 수 있는데, 이는 고윳값을 대각선 성분으로 갖는 diagonal matrix 가 된다. 다시 써보면,
    
    <p>
    $$
    V^{-1}AV = \\ 
    \begin{bmatrix} 
    \lambda_1 &  &... &  & 0\\
    . & \lambda_2 & &  &. \\
    . & & \lambda_3 &   &. \\
    . &  &  & \lambda_4 &. \\
    0 &  & ... &  &\lambda_5 \\
    \end{bmatrix} 
    $$
    </p>
    
    이다. change of basis 의 관점에서 살펴보면, 이는 $V$ span 에서의 $A$ 변환이다. 그런데 $V$는 basis가 eigenvector 인 변환이므로, $A$에서 오직 scaling 하기만 하는 coordinate system 으로 변환된 것이다. 즉 diagonal matrix 일 수밖에 없고, 이 scale 하는 정도가 eigenvalue 로 나타내어지기 때문에 위 quadratic form 은 이렇게 eigenvalue 로 이루어진 diagonal matrix를 갖게 된다. 
    
- 이를 통해 역으로 $A$ 를 구성해보면,
    
    $$
    A = V diag(\lambda ) V^{-1}
    $$
    
    라는 우리가 알고 있는 eigend-decomposition 을 이끌어낼 수 있다. 즉 eigen-decomposition은 선형변환을 통해서 $A$라는 span에 구애받지 않는 회전축을 구하고, 이를 통해 공간을 원하는 방향으로 확장할 수 있게 해주는 도구이다. 간단한 예제로써 선형 변환의 반복적인 적용 작업에 이를 많이 이용하는데, quadratic form 의 앞뒤 형태가 서로 역행렬이기 때문이라는 수치해석적인 접근 말고도, 애초에 eigen-decomposition이 의미하는 바가 span에 구애받지 않고 scaling 이기 때문에 같은 해석이 가능하다. 
    
    또한 이는 PCA 등에서 covariance matrix를 maximize 하는 feature extraction 이 covariance matrix 의 eigenvalue 가 큰 eigenvector 를 선택하는 것과 일맥상통한다. eigenbasis를 통해서는 선형 변환 이후에도 점을 기술하는 방식이 바뀌지 않으므로 큰 기여도를 갖는 방향의 순서대로 basis를 선택하는 것을 PCA라고도 해석할 수 있다. 
    
- $A = V diag(\lambda ) V^{-1}$ 인 eigen-decomposition 으로 나타내었을 때, determinant의 성질에 따라 $det(A) = \Pi _i \lambda _i$ 임을 쉽게 알 수 있다. 따라서 만약 행렬이 nonsingular matrix 라는 의미는, $\forall _i \lambda _i \neq 0$  이라는 말과 같다. 이를 통해서 postitive definite, semidefinite matrix 등이 정의되는데, 이는 span 이 축소되지 않으면서 basis 들의 순서가 바뀌지 않는 eigenvector 를 지니는 matrix 들을 총칭하는 의미가 된다.

### 3.3. **Similar Matrix**

$$
B = P^{-1}AP
$$

- nonsingular matrix P에 대해, A와 B를 위와같이 나타낼 수 있을 때 A와 B 를 서로 similar 하다고 정의한다. change of basis 관점에서 이는 같은 변환이지만 서로 다른 span에서의 표현이라고 알 수 있다. 또한 determinant 의 성질에 의해서 $det(B) = det(A)$ 임을 알 수 있다. 또한 다음을 통해 similar matrix 의 eigenvalue 값이 동일함을 유도할 수 있다.
    
    <p>
    $$
    B−λI=P^{−1}AP−λP^{−1}P=P^{−1}(A−λI)P \\ \therefore det(B−λI)=det(P^{−1}(A−λI)P)=det(A−λI)
    $$
    </p>
    
    이를 이용해 꽤 유용한 성질 하나를 유도할 수 있다. 다음을 보자 
    
    *"$C=P^TAP$ 를 만족하는 orthogonal matrix P가 존재할 때, A는 symmetric 이면 eigenvector 는 서로 orthogonal이다. "*
    
    <p>
    $$
    Av_j = \lambda _j v_j \\ \rightarrow v_i^T (Av_j) = (v_i^TA^T) v_j  = (Av_i)^T v_j \\ = \lambda _i \cdot v_i^T \cdot v_j \\ = \lambda _j \cdot v_i^T \cdot v_j
    $$
    </p>
    
    $\lambda_i \neq \lambda _j$  이기 때문에 어떤 eigenvector 도 서로의 내적값이 0 이 됨을 알 수 있다. orthogonal matrix 로 matrix factorization 을 하면 eigen-decomposition 에 더해서 특이한 성질을 하나 발견할 수 있는데, basis 들끼리 서로 orthogonal 하므로 직교하는 축들의 모임이라고 해석할 수 있고, 즉 이 orthogonal column matrix 자체를 회전변환이라고 해석할 수 있다. (원래의 basis 들도 서로 orthogoanal 한 관계를 이루므로)
    

### 3.4. **Singular Value Decomposition**

이제 단순히 eigen-decomposition 의 non-square matrix 으로 확장하는 대신, 위에서 상기한 orthogonal similar 개념을 통해서 SVD를 다시 생각해보자. 

- $A \in \textrm R^{m\times n}$ 인 matrix 를 생각하자. ($m \neq n)$ 이는 곧 span 이 collapse 하거나 expansion 하는 변환을 의미하고, 따라서 eigen-decomposition이 정의되지 않는다. 대신에 아래 두가지 symmetric matrix 를 구성할 수 있는데,
1. $A^TA$ : $n \times n$  symmetric matrix, 즉 eigenbasis 가 each orthogonal & independent 하며, basis 는 $A$의 column dimension을 rank로 갖는 를 span 으로 갖는다. 
2. $AA^T$ : $m \times m$ symmetric matrix 이다. span의 rank가 $A$ 의 row dimension 이 된다. 

각 symmetric matrix 의 의미를 잘 떠올리면서 정의에 따라 SVD 를 구성해보자. 

<p>
$$
A= U \Sigma V^T \\ U, V : \text{each orthonormal}
$$
</p>

이고, 이에 따라 $A^TA$ 와 $AA^T$ 를 나타내보면, 

<p>
$$
A^T A = V \Sigma ^T \Sigma V^T \\ A A^T = U \Sigma  \Sigma^T U^T 
$$
</p>

이다. 즉 $\Sigma ^T \Sigma$ 는 $A^TA$ 의 each orthogonal 한 eigenbasis 의 행렬이며, $U$ 는 $AA^T$ 에서 그렇다. 위에서 첫번째 식을 이용해 각각의 의미를 좀 더 살펴보도록 하겠다. 양변의 오른쪽에 행렬 $V$를 곱해보자. 

<p>
$$
A^TA v_j = \sigma _j^2 v_j  \ (A^TA V = V \Sigma^T \Sigma)
$$
</p>

1. $AA^T (Av_j ) = \sigma _j^2 (Av_j)$  : 
eigenvector 의 정의에 의해 $Av_j$  는 $AA^T$의  eigenvector 이다. (이때, $v_j$는 $V$의 한 column인 $A^TA$의 eigenvector 중 하나이다)

2. $v_j^T A^TAv_j = \sigma_j^2 v_j^T v_j \quad \rightarrow \quad (Av_j)^T (Av_j) = \sigma_j^2$
즉 $Av / \sigma$ 는 $AA^T$의 unit eigenvector $u$ 이다. ($u \in U$ ) 따라서, 

<p>
$$
Av_j= \sigma _j u_j \\ \rightarrow AV = U \Sigma \\  \rightarrow A = U \Sigma V^T  
$$
</p>

위를 통해 SVD 가 non square matrix 로부터 비롯되는 symmetric square matrix $A^TA$를 통해서 정의됨을 알 수 있다. 이는 기하학적으로도 eigen-decomposition과 유사한 의미를 갖는데, $U, V$ 가 모두 each orthonormal 한 column 으로 이루어진 square matrix 이기 때문에 이는 회전변환의 의미를, $\Sigma$는 $A^TA$의 eigenvalue 를 diagonal element 로 갖는 scaling 변환의 의미를 갖는다. $U$와 $V$는 서로  **dimension 이 다를 수 있지만**, 둘 다 회전 변환이라는 본질적인 성질을 공유한다.

### 3.5. **Moore-Penrose pseudo-inverse Matrix**

유도 과정을 자세하게 보자.

$$
Ax = b
$$

라는 선형방정식에서, $A$ 가 non square 혹은 non singular matrix 일 경우에는 역행렬을 통한 일반해를 구할 수 없다. 따라서 위의 SVD 에서 했던 것처럼, $A$를 통해 다루기 편한 symmetric 이고 square 인 matrix 를 만드는 것으로 이를 유도할 수 있다.

<p>
$$
A^TA x = A^T b \\ \rightarrow x = (A^T A)^{-1} A^T b
$$
</p>

<p>
$$
(A^TA)^{-1} A^T  = \{ (V \Sigma ^T U^T) (U \Sigma V^T) \}^{-1} V \Sigma^T U^T \\ = V (\Sigma ^T \Sigma )^{-1} \Sigma ^T U^T  \\ = V diag (\sigma ^{-1} ) U^T
$$
</p>

즉 pseudo inverse matrix 는 $A^TA$의 eigenvalue의 역수로 정의되는 값을 singular value 로 가지는 행렬이다. 
앞쪽에 등장하는식은 원래 least square solution 의 상황에서 L2 regularization 을 고려한 문제에서 비롯된 공식이다. 

<p>
$$
{d \over dx} {\| Ax - b\| ^2 _2 + \alpha \| x\| ^2 _2 }  \\ = (A^TA + \alpha I )x \\ \rightarrow \hat x =  (A^TA + \alpha I )^{-1} A^T b
$$
</p>

기술되어 있는 least square solution 이 $\min \|x \|_2$ 인 해를 갖는 이유를 기하학적으로 접근하는 방식도 있으나 그림을 그리기 어려운 관계로 이는 생략하겠다. 
고등 기하와벡터 정도의 내용으로 그리 어렵진 않다. (이젠 고등이 아닌...)

## 4. Trace Operator & Norm

### 4.1. Norm

$$
\|x \|_p = (\sum _i x_i ^p ) ^{1 \over p }
$$

<p>
$$
\| A {\|}_F = \sqrt{ \sum_{i, j} A_{i,j}^2}
$$
</p>

### 4.2. Trace Operator

$$
tr (A) = \sum_i A_{i,i}
$$

trace operator 에는 여러가지 흥미로운 성질들이 존재한다. 일부 유용한 성질들과 간단한 증명을 기술하겠다. 

1. Linearity : $tr(cA + dB) = c  \cdot tr (A) + d \cdot  tr (B)$

2. $tr(A) = tr(A^T) \quad$ (by definition)

3. Cyclic moving :  $tr(AB) = tr(BA)$

<p>
$$
\sum_i (AB)_{i,i} = \sum _i \sum _j a_{i,j} b_{j,i} = \sum_j \sum _i b_{j,i}a_{i,j} = \sum _j (BA) _{j,j}
$$
</p>

4. $tr(A)= \sum \lambda$  

<p>
$$
tr(A) = tr(V diag(\lambda)V^{-1} ) \\ = tr(V^{-1} V diag(\lambda)) \\ = tr(diag(\lambda)) \\  = \sum \lambda
$$
</p>

5. Frobenious Norm : $\|A\|_F ^2 = tr(AA^T) = tr(A^TA)$

<p>
$$
AV = U\Sigma \\ \rightarrow \| AV \|_F ^2= \| U\Sigma \| _F^2 \\ \rightarrow \|A \| _F ^2= \| \Sigma \| _F ^2 = \sum \sigma ^2
$$
</p>

<p>
$$
A^TA = V (\Sigma ^T \Sigma ) V^T \\ \therefore tr(A^TA) = tr(AA^T) = \sum \sigma ^2 \\ \|A \|_F = \sqrt { tr(A^TA) }
$$
</p>

## 5. Principal Component Analysis

PCA 수식 유도 과정에서 lagrangian multiplier 를 사용하는 것이 일반적이기 때문에 후에 duality 를 다룬 이후에 다시 설명하도록 하겠다. 
대신에 자주 사용하는 covariance matrix 에 대해 간단하게 기술하고, lagrangian multiplier 외의 방법으로 PCA와 eigen-decomposition 간의 관계를 살펴보자. 

### 5.1. Covariance Matrix

$$
cov(A) = {1 \over n} (A - \vec \mu ) ^T (A- \vec  \mu)
$$

위 식에서 $A- \vec \mu = X$ 라 하자. 그렇다면, 이를 통해 eigen-decomposition 이 가능하다. 

<p>
$$
X^TX = V^{-1} \Sigma V \\ = V^T \Sigma V^{-T} \quad (\because X^TX : \text {symmetric}) \\ \rightarrow VV^T \Sigma (V^TV) ^{-1} = \Sigma \\ \therefore VV^T = I 
$$
</p>

따라서 위 등식이 성립하고, 이는 즉 covariance matrix 의 eigenbasis 는 서로 orthogonal 하다는 의미이다. covariance matrix 를 이루는 basis 가 서로 내적값이 0이므로, 이를 uncorrelated 되었다고도 한다. 

PCA 에서 이를 최대화 하는 방향으로 feature extraction 을 진행하는 것은, data 를 통해 구축한 covariance matrix는 data 가 얼만큼 퍼져있는지를 나타내고 있기 때문이다. 즉 분산을 최대한 보존하는 방향으로 진행해야 dimension reduction 이 일어나도 정보의 손실이 최소화 되는 것이다 (lessy compression). 

### 5.2. Principal Component Analysis :

encoder matrix 의 최종 derivation form 인 아래 식을 통해서 PCA 와 eigen-decomposition 간의 관계를 살펴볼 수 있다. DL book의 유도에 따르면 PCA는 곧,

<p>
$$
\argmax _d \ Tr(d^T X^TXd) \quad \text{where } d^T d = 1
$$
</p>

이라는 형태가 나오는데,  $d^TX^TXd$ 는 $X^TX$ induced ellipsoid 임을 간단하게 알 수 있다.
수식에서 간단한 첨언을 하면, 제약조건으로써 제시하고 있는 $\|D \|_F = I$ 이라는 수식은 maximization 문제를 풀 때, 단순히 D 행렬의 크기만을 키우면 objective function 값이 커지기 때문에 이 값에 제약을 둔 것이다.  

$X^TX$ 는 symmetric matrix 이므로,  $X ^TX = P^T AP$ 를 만족하는 eigendecomposition diagonal matrix $A$ 를 만들 수 있다. 따라서, $Pd' = d$  인 $d'$  에 대하여,  

<p>
$$
d^TX^TXd = (Pd')^T X^TX (Pd') \quad  \quad \quad  \\  = d'^T (P^TX^TXP)d' 
$$
</p>

를 만족하고, $P^TX^TXP$ 는 eigenvalue 를 diagonal element 로 가지는 matrix 가 된다. 즉 eigenbasis 로 이루어지는 span 상의 ellipsoid plane 을 의미하고, 이 ellipse 를 이루는 axis 중에 가장 큰 것을 고른다는 의미가 된다. 구체적으로 ellipse 식을 구해보자면, 

<p>
$$
\begin{bmatrix} 
a_1 &  &... &  & a_n \\
\end{bmatrix} 
\begin{bmatrix} 
\lambda_1 &  &... &  & 0\\
. &  .& &  &. \\
. & &  .&   &. \\
. &  &  & . &. \\
0 &  & ... &  & \lambda _n\\
\end{bmatrix} 
\begin{bmatrix} 
a_1 \\
. \\
. \\
. \\
a_n \\
\end{bmatrix}  \\ = \lambda_1 (a_1)^2 + \lambda _2 (a_2)^2 + ... + \lambda_n (a_n)^2 \\ \text{where } d'^T = \begin{bmatrix} 
a_1 &  &... &  & a_n \\
\end{bmatrix} 
$$
</p>

즉, eigenbasis 로 이루어진 coordinate system 의 ellipse 임을 확인할 수 있다. 

기하학적인 설명 이외에 $d$ 가 eigenvector 일 때, eigenvalue 가 max 값인 이유또한 수식적으로 간단하게 유도해볼 수 있다. 다음을 보자. 

<p>
$$
\ d^T X^TX d  \quad \text{where } d^T d = 1 \\ = d^T (\lambda d ) \\ = \lambda \  d^Td \ \ \\ = \lambda  
\quad 
$$
</p>

따라서 $d$ 가 eigenvector 일 때, 해당 식의 값이 eigenvalue 값과 동일하고, 따라서 최대 eigenvalue 에 해당하는 eigenvector 를 찾는 것이 PCA 임을 확인할 수 있다. 즉 $cov(X^TX)$  의 최대 eigenvalue 에 해당하는 eigenvector 를 찾는 작업이 곧 PCA 이다. 

또한 이를 eigen-decomposition 의 정의에서 오는 직관을 통해 유추해볼 수 있는데, 큰 값의 eigenvalue 는 data 를 기술하는데 큰 기여를 하는 axis 이고 (scaling 의 민감도가 높기 때문에), 따라서 이것의 크기 순서대로 보존하는 것이 전체 data 를 이루는데 많은 기여를 하는 axis 들만 남겨놓는 것이라고도 해석할 수 있다.