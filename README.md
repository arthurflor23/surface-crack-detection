## Surface Crack Detection and Segmentation

Digital Image Processing class project, Master's degree, UPE - POLI (2018.2).

This project aims to detect and segment surface cracks in images using Digital Image Processing and deep learning techniques. The U-Net model was used for both crack detection and segmentation processes. For more details on the U-Net model, see the [paper](https://arxiv.org/abs/1505.04597v1) and the [repository](https://github.com/zhixuhao/unet).

### Results

#### Cracked Tile Detection

| Original  | Predicted | Overlay |
| ------------- | ------------- | ------------- |
|   <img src="https://github.com/arthurflor23/cracked-tile-detection/blob/master/out/cracktile/001_3_original.png?raw=true" width="275"/>  | <img src="https://github.com/arthurflor23/cracked-tile-detection/blob/master/out/cracktile/001_2_predict.png?raw=true" width="275" />  | <img src="https://github.com/arthurflor23/cracked-tile-detection/blob/master/out/cracktile/001_4_overlay.png" width="275" /> |

#### Cracked Concrete Detection

| Original  | Predicted | Overlay |
| ------------- | ------------- | ------------- |
|   <img src="https://github.com/arthurflor23/cracked-tile-detection/blob/master/out/crackconcrete/001_3_original.png?raw=true" width="275"/>  | <img src="https://github.com/arthurflor23/cracked-tile-detection/blob/master/out/crackconcrete/001_2_predict.png?raw=true" width="275" />  | <img src="https://github.com/arthurflor23/cracked-tile-detection/blob/master/out/crackconcrete/001_4_overlay.png?raw=true" width="275" /> |

### Citation

If this project has been helpful for your research, please consider citing the following paper:

```
@article{10.1590/s1678-86212021000100498,
   title     = {{Processamento digital de imagens para detec\~A\S\~A\poundso autom\~A!`tica de fissuras em revestimentos cer\~A\textcentmicos de edif\~A\-cios}},
   journal   = {{Ambiente Constru\~A\-do}},
   author    = {Ruiz, Ramiro Daniel Ballesteros AND Lordsleem Junior, Alberto Casado AND Neto, Arthur Flor de Sousa AND Fernandes, Bruno Jos\~A\copyright Torres},
   pages     = {139 - 147},
   volume    = {21},
   month     = {01},
   year      = {2021},
   publisher = {scielo},
   isbn      = {1678-8621},
   url       = {http://www.scielo.br/scielo.php?script=sci_arttext&pid=S1678-86212021000100139&nrm=iso},
   doi       = {10.1590/s1678-86212021000100498},
}
```
