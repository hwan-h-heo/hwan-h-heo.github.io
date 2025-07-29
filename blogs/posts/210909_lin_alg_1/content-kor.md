title: Linear Algebra for Deep Learning (1)
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

## 1. Linear System

### 1.1.1. Linear Combination

$$
\sum_i c_i v_i \\ c_i : \text{scalar}, \ v_i : \text{vector}
$$

- basis : span 을 이루는 기본 단위. 어떠한 좌표계를 정의하는 기본 단위이다.
- "*Span"* 에 대한 생각 : basis 를 조합하여 만들 수 있는 공간. 즉 basis 의 weighted sum 으로 표현할 수 있는 모든 점들의 집합. 가장 기초적인 연산만을 통해서 어떤 공간을 얻을 수 있는지에 대한 물음이라고 생각할 수 있다. 
ex) 2차원 직교 좌표계는, $(1,0) : i, (0,1) : j$   을 basis 로 하는 span 이라고 생각할 수 있다. 
$(5,2) = 5i + 2j$

↔ 만약 (1,0) , (2,0) 이었다면? (즉 basis 가 같은 line 에 존재) : 
(2,0) 이 (1,0) 을 통해서 만들 수 있다. 이 경우에 span 은 x 축이 된다. 따라서 단순히 basis 의 수에 따라서 span 의 dimension 이 정해지는 것이 아니고 basis를 어떻게 설정하는지가 중요할 것이라는 직관을 얻기가 가능하다.
- 이를 통해 span 의 rank가 정의되는데, span 을 정의하는 basis (즉 column) 들이 이루는 공간, column space의 dimension 수가 곧 rank 이다. full rank 의 의미는, span 에 어떠한 basis 를 추가하여도 rank 가 변하지 않을 때가 곧 full rank가 된다.

### 1.1.2. Linear Independent

$$
\sum_i c_i v_i \\ c_i : \text{scalar}, \ v_i : \text{vector}
$$

- 위의 linear combination 값이 $\forall _i \ c_i$ 가 0 일때만 0의 값을 가진다면 이를 linear independent 라고 하고, 아닐 경우엔 linear dependent 라고 한다. 

즉 어떤 vector 과 주어진 vector 들과 linear dependent 한지 판별하려면, vector set 의 weighted sum 으로 해당 vector 가 만들어지지 않을 때 이를 independent 라고 한다.
- 위 개념을 span 과 함께 다시 설명해보자면, span 에서 어떤 basis 를 빼도 span 의 dimension 이 축소되지 않는 경우에는 해당 vector 와 다른 basis vector 가 dependent 한 관계를 가진다고 해석할 수 있다. 즉 위 Span의 예제의 경우에서 
(1,0), (0,1) : 서로가 linear independent 
(1,0), (2,0) : 서로가 dependent 한 상황이 되는 것이다.

- 달리 말하면 서로 independent 한 basis에 대해서, 어떠한 vector 를 추가하였을 때, span 의 dimension 변화가 일어난다면 이는 기존 vector 들과 independent 한 vector 가 된다.
    
    ex) $(1,0,0) , (0,1,0), (0,0,1)$
    
    기존 벡터들의 Span에 포함되지 않는, 즉 **선형 독립인 새로운 벡터를 basis에 추가하면** Span의 차원이 확장되므로 **Rank가 늘어잉다.** 반대로, 기존 벡터들의 Span에 이미 포함되어 있는, 즉 **선형 종속인 벡터를 추가하면** Span에 변화가 없으므로 **Rank는 변하지 않는다.**
    

## 2. Linear Transformation

**Definition:** $L(ax+by) = a\ L(x) +b\ L(y)$

- *line → line*
- *origin → origin*

### 2.1. **Linear Transformation with the change of basis**

- 간단한 아래 예제를 보자.
    
    <p>
    $$     \begin{bmatrix}      3 & -2 \\     1 & 2 \\     \end{bmatrix}      =     \begin{bmatrix}      3 & -2 \\     1 & 2 \\     \end{bmatrix}      \begin{bmatrix}      1 & 0 \\     0 & 1 \\     \end{bmatrix}      $$
    </p>

    
    i, j 의 basis 가 $(3,1) : i', (-2,2) :j'$ 의 새로운 basis 로 변화되었다고도 볼 수 있다. 바뀐 basis 로 span을 구성했을 때도 여전히 2차원 직교 좌표계인데, 이는 (3,1), (-2,2) 가 서로 independent 하기 때문이다. (같은 line 위에 있지 않다)
    
    이때 $(5,2)$ 라는 점을 생각해보면, $(5,2)$ 는 위 변환을 통해 $(11,9)$ 라는 점으로 이동한다. 
    
    $$
    (11,9) = 5 i' + 2j' 
    $$
    
    즉 basis 만 이동하면, 원래 점을 기술하던 방식은 바뀌지 않는다. basis 를 어떻게 scaling 하여 점을 이루는지에 대한 정보를 보존하면서, 그 basis 만 바꾸는 변환을 linear transformation 이라고 할 수 있을 것이다. 
    
    Determinant 가 0인 경우에는, 어떤 matrix 로 정의되는 linear transformation 이후의 바뀐 두 basis 가 서로 dependent 한 관계를 가지게 된다. 
    즉 determinant 가 0인 변환은 span 의 dimension 을 축소한다. 이에 따라서 역행렬이 가지는 의미도 생각해볼 수 있다. 
    
- 다음 예제를 보자.
    
    <p>
    $$
    \begin{bmatrix} 
    2 & 4 \\
    3 & 6 \\
    \end{bmatrix} 
    $$
    </p>
    
    위 변환은 $(1,0), (0,1)$ 의 두 basis 를 $(4,3), (6,2)$ 인 두 점으로 basis transformation 하는 변환이다.  위 두 점은 같은 line 위에 있는 점이고 $(y = {3 \over 2} x )$, span 전체의 transformation 은 위 line 하나로 국한된다. 
    
    즉 span 의 dimension 을 축소시키기 때문에, 바뀐 좌표계에서 원래 좌표계로의 대응이 일대일 대응이 아니게 된다. 따라서 바뀐 span 을 원래의 span으로 변환하는 linear transformation 을 정의할 수 없게 되고, 즉 역행렬을 정의할 수 없다.  
    
    행렬 $M$ 을 통해서 변화된 공간을 $M'$ 이라 할 때,  $rank(M') \le rank(M)$ 이 항상 성립한다. Span의 dimension은 유지하거나 줄어들기 때문.
    

### 2.2. **Non-square matrix case**

**2.2.1.**  

<p>
$$
\begin{bmatrix} 
a & b \\
c & d   \\
e & f   \\
\end{bmatrix} 
$$
</p>

- 3x2 행렬로 column space 가 평면을 이루고 있다. (서로 independent 라고 가정하자). 여기서 주의해야할 점은, 위 변환을 통해 2차원 상의 점이 3차원 상의 점으로 변환되지만, 그 점들이 이루는 공간은 3차원 상의 평면이라는 점이다. 

- 단순히 span 의 차원이 2라는 것과는 조금 다른 의미가 된다. Column space가 2차 직교 좌표계로 축소되는것이 아닌, 3차원 상의 hyperplane이 되는 것이다.

**2.2.2.**
<p>
$$
\begin{bmatrix}
a & b & c\\
d & e & f   \\
\end{bmatrix} 
$$
</p>

- 3차원 상의 점을 2차원의 점으로 사상하는 변환인데, basis vector 가 3개 이지만, column space 자체가 2차원이기 때문에 full rank가 아니다. 즉 span 이 축소되며, 3차원에서 2차원 위의 점으로 사영된다는 의미로 해석할 수 있다.  다시 말해 $m \times n$ matrix 에서 $m < n$ 인 경우에, 이 변환은 고차원 상의 점을 저차원으로 사영하는 변환이 된다. 

- 이를 통해서 dot product도 linear transformation 관점에서 해석할 수 있다. 어떤 두 벡터 $u, v$ 간의 내적 $u \cdot v$ 는, $v$ 벡터에 $u^T$  로 정의되는 linear transformation 을 행하는 것이라고 해석할 수 있다. 

- 이때 $u^T$  는 $1 \times n$ matrix 이고, basis 가 n개 이지만, 서로 모두 dependent 하다. 즉, $u^T$  가 이루는 span 은 dimension 이 1이기 때문에, linear projection 이라는 의미를 지니게 된다. 1차원은 scalar 이기 때문에, 내적값이 상수로 나오게 되는 것과 일맥상통한다.

### 2.3. **Matrix Multiplication**

- Matrix multiplication을 sequence of transformation 으로 생각하는 것 또한 가능해진다. 
직관적으로 행렬곱에 대해 commutative가 성립하지 않고 associative, distributive 는 성립하는 이유를 생각해볼 수 있다.
    
    $$
    A(B+C) = AB + AC
    $$
    
    Distributive 에 대해서는, A 또한 linear transformation 이기 때문에 linear transformation 의 성질에 따라 위 성질을 만족한다. 
    
    $$
    A(BC) = (AB)C
    $$
    
    Associative 는 결국 계산의 문제일 뿐, 세 변환을 순서대로 같은 순서로 적용하는 것과 같아진다. 
    
    $$
    AB \neq BA
    $$
    
    commutative 는 변환을 적용하는 순서가 달라지면 그 결과가 달라진다는 의미가 된다. 또한 transpose 에 대한 직관도 얻을 수 있는데, **어떤 선형 변환의 transpose 는 i 와 j 에 대해 작용하던 변환이 서로 반대로 적용된다**는 의미이다. 
    

 

### 2.4. **Determinant**

- 위에서 잠시 얘기했지만, 단순히 determinant 가 0인 경우를 제외하고, determinant 가 무엇을 의미하는지 생각해볼 필요가 있다. 간단한 예제를 통해서 살펴보자.
    
    2차원 좌표평면에 $(0,0), (0,1), (1,0), (1,1)$ 로 이루어진 크기 1 짜리 사각형이 있다. 위 사각형을 아래 변환을 통해서 linear transformation 하게 된다면,
    <p>
    $$
    \begin{bmatrix} 
    a & c \\
    b & d \\
    \end{bmatrix} 
    $$
    </p>
    
    원래 정의된 영역은 아래 그림에서 빨간 마름모꼴의 영역이 되는데, 그림을 바탕으로 해당 영역의 넓이를 구해보면, $$(a+c)(b+d) - ab - cd - 2 bc = ad - bc$$ 이 된다.
    
    즉 **determinant 의 계산적인 의미는, 원래 span 에서 의미하는 넓이가 얼마만큼 바뀌게 되는지에 관한 값** 이 된다. 
    

<img src='./210909_lin_alg/assets/image.png'>

- 따라서 **역행렬이 없는 determinant 가 0인 경우에는, 영역의 넓이를 보존하지 않는다** 로도 해석할 수 있다. 3차원 이상의 dimension 에서는 마찬가지로 영역의 부피를 scaling 하는 값으로 해석된다. 또한 determinant 값이 음수라는 의미는,  basis 방향이 바뀌었다는 의미로 해석할 수 있다.

- 또한 $det(M_1 M_2) = det(M_1) \cdot det(M_2)$ 로 계산할 수 있는 것은, 단순히 수치적인 의미가 아니라 sequence of transformation 이기 때문으로 해석 가능하다. 위 공식은 변환 $M_1M_2$ 가 $M_1$  과 $M_2$ 가 기존 span 의 넓이를 바꾸는 정도의 곱만큼 바뀐다는 의미가 된다. 

### 2.5. **System of linear equations**

$$
Ax =b
$$

위와 같은 형태의 선형방정식계의 경우엔, 앞서 말한 span 과 determinant 를 통해 면밀하게 살펴보는 것이 가능하다. 

1. $det(A) = 0$ (span 이 축소) : 축소된 span 에 b 가 존재하는지에 대한 물음이 된다. 
3차원 span 에서 3차원 위의 평면으로 span 이 축소된 경우에는 b vector가 해당 평면 위에 존재하는지에 대한 방정식이 될 것이다. 
이 경우엔 행렬 A의 column 으로 정의되는 column space 인 span 이 full rank 가 아니라는 의미가 된다. 이를 통해 null space 의 정의를 떠올려보면, span 의 dimension 이 축소되면서 원래 span 에서 origin 으로 축소된 모든 영역이 null space 가 된다. 즉 $Ax = 0$ 의 모든 해집합은 null space 이다. 

2. $det(A) \neq 0$ : 일대일 대응 변환이고, 역행렬이 존재하기 때문에 단일해가 존재한다. 해는, 
$x = A^{-1}b$ 가 될 것이다. 
Determinant를 통해 square matrix 인 경우 선형방정식계의 해를 구할 수 있는 Cramer's rule 을 기하학적으로 해석하는 것이 가능하다. 아래 그림을 보자. 

<img src='./210909_lin_alg/assets/image-1.png'>

위의 점을 $(x,y)$ 라고 하자, 그렇다면, x와 y의 값을 넓이로 가지는 영역은 아래와 같이 그릴 수 있다. 

<img src='./210909_lin_alg/assets/image-2.png'>

각각 $(1,0), (x,y)$ 를 변으로 갖는 넓이가 y인 영역, $(x,y), (0,1)$을 변으로 갖는 넓이가 x인 영역이다. 이 영역을 어떤 행렬 A를 통해 linear transformation 하면, 각각 $det(A) \cdot y , \ det(A) \cdot x$ 의 넓이를 가지게 된다 (determinan 의 성질). 이는 곧 각각 $i' , (x,y)$ 과 $(x,y), j'$ 이 이루는 영역의 넓이가 된다 (아래 그림).  즉, y 와 x 가 의미하는 바는 변환된 basis에서 $i', \ j'$ 을 $(x,y)$ 로 각각 대체하였을 때의 넓이의 비가 된다.

<img src='./210909_lin_alg/assets/image-3.png'>

예를 들어 아래와 같은 2차 정사각 행렬의 선형방정식계가 있을 때, 

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
\alpha  \\
\beta  \\
\end{bmatrix} 
$$
</p>

각각의 column 을 차례대로 $(x,y)$ 인 변환으로 대체하였을 때의 determinant 값을 이용하여 편리하게 위 linear system 의 해를 구할 수 있게 된다. 아래 방식은 전산적으로 linear equations 의 해답을 구할 때 이용하게 된다. 
<p>
$$
y = { det{\begin{bmatrix} 
a & \alpha \\
b & \beta \\
\end{bmatrix} } \over det{\begin{bmatrix} 
a & c \\
b & d \\
\end{bmatrix} }}, 
\ x = { det{\begin{bmatrix} 
\alpha & c \\
\beta & d \\
\end{bmatrix} } \over det{\begin{bmatrix} 
a & c \\
b & d \\
\end{bmatrix} }}, 
$$
</p>

차원이 확장돼도 마찬가지로 부피를 변화시키는 양으로 determinant 가 정의되기 때문에, 동일한 방식으로 고차원에서도 linear equations 의 해를 구할 수 있다. 또한 역행렬이 존재하지 않을 때 determinant 값이 0이 되어 분수에서 분모가 0이 될 수 없으므로 마찬가지로 해가 존재하지 않음을 알 수 있다.  

---

- [다음 글](./?id=210909_lin_alg_2)에 이어서...