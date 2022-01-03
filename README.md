## Table of contents
- [Project Description](#project-description)
    - [ADMM-ADAM workflow](#admm-adam-workflow)
- [Project Setup](#project-setup)
- [How To Restore Images In this Project](#how-to-restore-images-in-this-project)
- [Reference](#reference)
## Project Description
ADMM-ADAM is a powerful hyperspectral image (HSI) restoration framework designed by Dr. Chia-Hsiang in 2021 ([Journal Link](https://ieeexplore.ieee.org/document/9546991)). It's based on "convex optimization", "feature extractor" and "ADMM optimizer" to make a high reducibility image.

### ADMM-ADAM workflow
ADMM-ADAM workflow as follows:
![](https://i.imgur.com/ohtsqcl.png)

1. Train a Deep Learning model with ADAM optimizer (GAN in here), and get a Deep Learning olution (image)
2. Extract the most important information of Deep Learning solution by PCA (or some other feature extractor)
3. design a convex optimization problem with ADMM optimizer as follows:

    ![](https://i.imgur.com/92szBja.png)
  
    where, 
    - X:
        - ![](https://i.imgur.com/4IWOC7j.png) be an M-band hyperspectral (target) image with L pixels
    - Y:
        - ![](https://i.imgur.com/ROvSTS1.png) be the observed image, meaning that some of its entries are missing
    - Ω:
        - ![](https://i.imgur.com/n20bmqx.png) denote the index set of those available data.
    - X_{DL}:
        - ![](https://i.imgur.com/DKiYV9p.png) can be obtained using ADAM optimizer (GAN model with ADAM optimizer in here, you can get it by [darkfanxing/GAN](https://github.com/darkfanxing/GAN))
    - λ:
        - ![](https://i.imgur.com/kNyib67.png) is called regularization parameter empirically set as 0.01
          in this work
    - Q:
        - Q-qudratic norm, which extracts useful features from ![](https://i.imgur.com/0cCBpeI.png) for effective regularization
        - feature extractor is PCA in here

    Assume we have N materials, each pixel can be modeled as a linear combination of N spectral signature vectors in R^{M}. In other words, all the hyperspectral pixel vectors belong to a N-dimensional subspace if we ignore some non-linearity or noise effects, so the target image X can be represented as follows:

    ![](https://i.imgur.com/nfHmnDe.png)

    where,
    - ![](https://i.imgur.com/uMZ4MrQ.png): the most important N component (eigenvector), i.e. N material
    - ![](https://i.imgur.com/yEfBDVA.png): some coefficient matrix ![](https://i.imgur.com/jR0iJTS.png) and we can simplify the objective function:
        
    ![image](https://user-images.githubusercontent.com/36408071/147899750-21ff954b-0f97-432d-987c-3f1c36e23604.png)


    so convex optimization problem can be represented as follows:
    
        ![](https://i.imgur.com/OOxySYH.png)

    the meaning of F-norm is:
    
        ![](https://i.imgur.com/DwhQawt.png)
    
    
    Once ![](https://i.imgur.com/ng4maRj.png) is available, it can be used to reconstruct the
    complete hyperspectral image as ![](https://i.imgur.com/u3X2BWG.png)
    Hehe... it is happy time for reformulating convex optimization
    problem into the standard ADMM form:
        
    ![](https://i.imgur.com/VcQZcYQ.png)
    
    
    and give a augmented Lagrangian term ![](https://i.imgur.com/cDteZWp.png):
        
    ![](https://i.imgur.com/3BJtlVI.png)
  
    
    where,
    - ![](https://i.imgur.com/TEfQB7P.png):
        - ![](https://i.imgur.com/O53gNRc.png) is the scaled dual variable
    - ![](https://i.imgur.com/oncQMDl.png)
        - ![](https://i.imgur.com/oncQMDl.png) is the penalty parameter, empirically set as 0.001
    
    
    Then, ADMM optimizer solves the problem as detailed as follows:
    
    ![](https://user-images.githubusercontent.com/36408071/147899226-87c441d5-944f-469a-bf69-5b19c0d92450.png)
    
    
    where,
    - ![](https://user-images.githubusercontent.com/36408071/147899506-f9be9971-d318-4f66-96ae-7b3d52ea44b7.png)
    - ![](https://user-images.githubusercontent.com/36408071/147899421-f2016cab-c6f8-4b10-99af-854e7b76d1e0.png)
    - ![](https://user-images.githubusercontent.com/36408071/147899456-24c83158-ea4a-46d7-b608-a51301cbbb91.png)

    where,
    - ![](https://user-images.githubusercontent.com/36408071/147899603-c6daf91a-ef83-4da3-bab0-28d3bdb1b8cd.png)   
        - ![](https://user-images.githubusercontent.com/36408071/147899590-70eae922-9ae6-4384-80f4-06d297fbbbac.png)
    - ![](https://user-images.githubusercontent.com/36408071/147899616-6c9d60d1-33ae-4f55-b053-7a3980957d10.png)
        - ![](https://user-images.githubusercontent.com/36408071/147899666-52178a14-d259-41f5-97e9-0dda87d8b38a.png)


## Project Setup
To avoid Python package version conflicts, the project use pipenv (Python vitural environment) to install Python packages.

```console
pip install pipenv
pipenv shell
pipenv install
```

## How To Restore Images In This Project
```console
python src/main.py
```

## Reference
Lin, Chia-Hsiang, Yen-Cheng Lin, and Po-Wei Tang. "ADMM-ADAM: A New Inverse Imaging Framework Blending the Advantages of Convex Optimization and Deep Learning." IEEE Transactions on Geoscience and Remote Sensing (2021).
