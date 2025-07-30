const postsData = [
    {
        id: '250710_building_large_3d_2',
        title_eng: 'Building Large 3D Generative Model (2) VAE and DiT Deep Dive',
        title_kor: 'Large 3D Generative Model 구축하기 (2) VAE and DiT 톺아보기',
        subtitle_eng: 'VAE and DiT Architecture for Vecset vs. Sparse-Voxel',
        subtitle_kor: 'VAE and DiT 구조 분석 for Vecset vs. Sparse-Voxel',
        date: '2025-07-10',
        category: 'post', // 'post' 또는 'note'
        series: '3d-generation',   // 시리즈가 있으면 시리즈 ID, 없으면 null
        languages: ['kor']
    },
    {
        id: '250702_building_large_3d_1',
        title_eng: 'Building Large 3D Generative Model (1) Data pre-processing',
        title_kor: 'Large 3D Generative Model 구축하기 (1) Data pre-processing',
        subtitle_eng: 'Watertight Mesh Conversion and Salient Edge Sampling',
        subtitle_kor: 'Watertight Mesh 변환과 Salient Edge Sampling',
        date: '2025-07-02',
        category: 'post', // 'post' 또는 'note'
        series: '3d-generation',   // 시리즈가 있으면 시리즈 ID, 없으면 null
        languages: ['kor']
    },
    {
        id: '250310_model_viewer',
        title_eng: '3D Model Viewer in Web',
        title_kor: '커스텀 3D 웹뷰어 만들기',
        subtitle_eng: 'Custom Threejs 3D Model Viewer',
        subtitle_kor: '커스텀 Three.js 3D 모델 뷰어',
        date: '2025-03-10',
        category: 'post', // 'post' 또는 'note'
        series: 'web-3d',   // 시리즈가 있으면 시리즈 ID, 없으면 null
        languages: ['eng', 'kor']
    },
    {
        id: '250302_3d_latent_diffusion',
        title_eng: 'An Era of 3D Generation',
        title_kor: '3D 생성 모델의 시대',
        subtitle_eng: 'From ShapeVAE to Trellis and Hunyuan3D',
        subtitle_kor: 'ShapeVAE 부터 Trellis 와 Hunyuan3D 까지, 최근 3D 생성 트렌드의 모든 것',
        date: '2025-03-02',
        category: 'post',
        series: '3d-generation',
        languages: ['eng'],
    },
    {
        id: '250106_tomography',
        title_eng: 'Neural Rendering Beyond Photography',
        title_kor: 'X-Ray 와 NeRF',
        subtitle_eng: 'NeRF for Tomography & Tip for NeRF Viewer',
        subtitle_kor: '의료데이터를 위한 NeRF modeling 과 NeRF Viewer 팁',
        date: '2025-01-06',
        category: 'post',
        series: 'nerf-and-gs',
        languages: ['eng', 'kor']
    },
    {
        id: '240917_3djs',
        title_eng: 'Add Gaussian Splatting to Your Website',
        title_kor: '웹사이트에 3D GS 삽입하기',
        subtitle_eng: 'Tutorial for GS Scene w/ Three-js',
        subtitle_kor: 'Three-js 이용한 custom GS scene 웹뷰어 tutorial',
        date: '2024-09-17',
        category: 'post',
        series: 'web-3d',
        langugaes: ['eng', 'kor'],
    },
    {
        id: '240823_grt',
        title_eng: 'Don\'t Rasterize, But Ray Trace 3D Gaussian',
        title_kor: '3D Gaussian Ray Tracing 톺아보기',
        subtitle_eng: 'In-Depth Review of 3D Gaussian Ray Tracing by NVIDIA',
        subtitle_kor: '3D Gaussian Ray Tracing 분석하기',
        date: '2024-08-23',
        category: 'post', 
        series: 'nerf-and-gs',
        languages: ['eng', 'kor']
    },
    {
        id: '240805_gs',
        title_eng: 'A Comprehensive Analysis of Gaussian Splatting Rasterization',
        title_kor: 'Gaussian Splatting Rasterization 완벽 분석',
        subtitle_eng: 'Understanding 3D GS\'s Rasterization Algorithm',
        subtitle_kor: 'GS rasterization 알고리즘과 cuda 구현 톺아보기',
        date: '2024-08-05',
        category: 'post', 
        series: 'nerf-and-gs',
        languages: ['eng', 'kor']
    },
    {
        id: '240721_sfm',
        title_eng: 'Radiance Fields from Deep-based Structure-from-Motion',
        subtitle_eng: 'Comparison of VGGSfM & MAsT3R',
        date: '2024-07-21',
        category: 'post', 
        series: 'nerf-and-gs',
        languages: ['eng',]
    },
    {
        id: '240602_2dgs',
        title_eng: 'Under the 3D: Geometrically Accurate 2D Gaussian Splatting',
        title_kor: '2D Gaussian Splatting 톺아보기',
        subtitle_eng: 'Understanding 2D GS\'s Algorithm',
        subtitle_kor: '3D GS 의 mesh recon 어려움과 이를 해결한 2D GS',
        date: '2024-06-02',
        category: 'post', 
        series: 'nerf-and-gs',
        languages: ['eng', 'kor']
    },
    {
        id: '240426_diffusion_depth',
        title_eng: 'Is Diffusion\'s Estimated Depth Really Good?',
        title_kor: 'Diffusion 으로 추정한 Depth 진짜 좋나요?',
        subtitle_eng: 'Making Mesh from Estimated Depth Map by Diffusion',
        subtitle_kor: 'Diffusion 으로 추정한 Depth Map 을 이용해 Textured Mesh 만들어보기 (feat. Marigold)',
        date: '2024-04-26',
        category: 'post', 
        series: '3d-generation',
        languages: ['eng', 'kor']
    },
    {
        id: '240226_sora',
        title_eng: 'Can Sora Understand 3D?',
        title_kor: 'Sora 가 상상하는 3D World',
        subtitle_eng: 'Radiance Fields Reconstruction from Video Generative AI',
        subtitle_kor: 'Video Generative AI 로부터 NeRF reconstruction 해보기',
        date: '2024-02-26',
        category: 'post', 
        series: '3d-generation',
        languages: ['eng', ]
    },
    {
        id: '231130_nerf_in_game',
        title_eng: 'Can NeRF be used in Game?',
        title_kor: 'NeRF 를 게임 제작에서 이용할 수 있을까?',
        subtitle_eng: 'Explore Limitations of NeRF',
        subtitle_kor: 'NeRF 의 단점을 살펴보자',
        date: '2023-11-30',
        category: 'post', 
        series: 'nerf-and-gs',
        languages: ['eng', ]
    },
    {
        id: '230202_ngp',
        title_eng: 'Instant-NGP Review & Re-Implementation',
        title_kor: 'Instant-NGP 리뷰 및 재구현',
        subtitle_eng: 'Review of Instant-NGP and PyTorch Re-Implementation',
        subtitle_kor: 'Review of Instant-NGP and PyTorch Re-Implementation',
        date: '2023-02-02',
        category: 'post', 
        series: 'nerf-and-gs',
        languages: ['eng', ]
    },
    {
        id: '211128_fourier',
        title_eng: 'Why Positional Encoding Makes NeRF more Powerful',
        title_kor: 'Position Encoding 이 NeRF 에서 필수적인 이유',
        subtitle_eng: 'Review of Fourier Features Let Networks Learn High-Frequency Functions',
        subtitle_kor: 'Fourier Features Let Networks Learn High-Frequency Functions 리뷰',
        date: '2021-11-28',
        category: 'post',
        series: 'nerf-and-gs',
        languages: ['eng']
    },
    {
        id: '210910_lin_alg_2',
        title_eng: 'Linear Algebra for Deeplearning (2): Change of Basis',
        title_kor: '딥러닝을 위한 선형대수 (2): Change of Basis',
        subtitle_eng: 'Diagonalization, Matrix Decomposition and Principal Component Analysis (PCA)',
        subtitle_kor: '대각화와 고윳값, 특이값 분해, Principal Component Analysis (PCA)',
        date: '2021-09-10',
        category: 'note',
        series: 'linear-algebra',
        languages: ['kor']
    }, 
    {
        id: '210909_lin_alg_1',
        title_eng: 'Linear Algebra for Deeplearning (1): Linear System',
        title_kor: '딥러닝을 위한 선형대수 (1): Linear System',
        subtitle_eng: 'Linear System, Linear Transformation, and its Intuitions',
        subtitle_kor: '선형시스템과 선형변환, 이에 대한 직관들',
        date: '2021-09-09',
        category: 'note',
        series: 'linear-algebra',
        languages: ['kor']
    }, 
    {
        id: '210602_cv4',
        title_eng: 'Computer Vision (4): 3D Stereo Vision and Epipolar Geometry',
        title_kor: '컴퓨터 비전 (4): 3D Stereo Vision and Epipolar Geometry',
        subtitle_eng: 'Camera Projection, Stereo, and Epipolar Geometry',
        subtitle_kor: 'Camera Projection, Stereo, and Epipolar Geometry',
        date: '2021-06-02',
        category: 'note',
        series: 'computer-vision',
        languages: ['kor']
    }, 
    {
        id: '210502_cv3',
        title_eng: 'Computer Vision (3): Homography and Image Alignment',
        title_kor: '컴퓨터 비전 (3): Homography and Image Alignment',
        subtitle_eng: 'Image Transformation, Homography, and Correspondence Matching',
        subtitle_kor: 'Image Transformation, Homography, and Correspondence Matching',
        date: '2021-05-02',
        category: 'note',
        series: 'computer-vision',
        languages: ['kor']
    }, 
    {
        id: '210402_cv2',
        title_eng: 'Computer Vision (2): Feature Extraction and Detection',
        title_kor: '컴퓨터 비전 (2): Feature Extraction and Detection',
        subtitle_eng: 'Image Gradient, Sobel Filter, LoG, and Detection',
        subtitle_kor: 'Image Gradient, Sobel Filter, LoG, and Detection',
        date: '2021-04-02',
        category: 'note',
        series: 'computer-vision',
        languages: ['kor']
    }, 
    {
        id: '210302_cv1',
        title_eng: 'Computer Vision (1): Image Filter and Morphology',
        title_kor: '컴퓨터 비전 (1): Image Filer and Morphology',
        subtitle_eng: 'Various Image Filters and Morphology (Erosion & Dilation)',
        subtitle_kor: '다양한 이미지 필터와 Morphology 연산: Erosion & Dilation 에 대하여',
        date: '2021-03-02',
        category: 'note',
        series: 'computer-vision',
        languages: ['kor']
    }, 
];

const seriesInfo = {
    'nerf-and-gs': {
        eng: 'Radiance Fields & Gaussian Splatting', 
        kor: 'Radiance Fields & Gaussian Splatting',
    },
    '3d-generation': {
        eng:'3D Generative AI',
        kor:'3D 생성 AI',
    },
    'web-3d': {
        eng:'3D in Web',
        kor:'웹에서 3D 구현하기'
    },
    'linear-algebra': {
        eng:'Linear Algrebra for Deeplearning',
        kor:'딥러닝을 위한 선형대수'
    },
    'computer-vision': {
        eng:'Classical Computer Vision',
        kor:'고전 Computer Vision'
    },
};