## 들어가며

3D Gaussian Splatting 의 최대 장점은 그 무엇보다도 100fps 이상의 렌더링 '속도' 이다. 
그 어떤 NeRF-based method 보다도 빠른 이 속도는 잘 설계된 tile-based rasterization 덕분인데, 오늘은 이 rasterization 관점에서 3D Gaussin Splatting 을 (최대한) 완벽하게 이해해보는 시간을 갖도록 해보자. 


## 1. 3D Gaussian as Primitive Kernel
![](https://miro.medium.com/v2/resize:fit:603/1*s6j7Hj9cg9Re9lxlqAHq5w.png)

3D Gaussian Splatting 의 기본 idea 는, implicit 한 NeRF 와 다르게 ***explicit*** 하고 학습 가능한 primitives 로 scene 을 표현하자는 것이다. (explicit 한 표현으로 갖는 여러가지 장점이 있는데, 이는 저번 2D GS review 에서 언급한 바 있으니 궁금하다면 [링크](https://velog.io/@gjghks950/Review-2D-Gaussian-Splatting-for-Geometrically-Accurate-Radiance-Fields-Viewer-%EA%B5%AC%ED%98%84-%EC%86%8C%EA%B0%9C#11-3d-gaussian-splatting)를 참조하기 바란다.)

저자들은 이러한 primitive (particle) 에 대한 kernel 을 다음과 같은 3D Gaussian function 으로 정의하였다. 

$$
G(x) = \exp \left( {- \frac{1}{2} x^{\rm T} \Sigma^{-1} x} \right )
$$