# SCTNet
This project provides the implementation of the Shallow CNN-Transformer Network (SCTNet), which achieve state-of-the-art performance in cloud detection tasks with 0.7M parameters and 1G MACs. Our code will be released soon.

## Quantitative Comparison (mIoU, %, ↑) on six datasets
| Method | HRC_WHU | GF1MS | GF2MS | CloudSEN12 L1C |CloudSEN12 L2A | L8 Biome|
|:-------|:--------:|:-------:|:--------:|:-------:|:-------:|:-------:|
|CDNetv1	|77.79|	81.82	|78.20|	60.50|	62.39|	34.58|
|CDNetv2|	76.75|	84.93	|78.84|	65.60	|66.05|	43.63|
|KappaMask|	67.48	|92.42|	72.00|	41.27|	45.28|	42.12|
|DBNet	|77.78	|91.36|	78.68	|65.52|	65.65|	51.41|
|SCNN|	57.22|	81.68	|76.99|	22.75	|28.76|	32.38|
|MCDNet	|53.50|	85.16|	78.36|	44.80|	46.52	|33.85|
|HRCloudNet|	83.44|	91.86|	75.57|	68.26	|68.35	|43.51|
|UNetMobv2	|79.91	|91.71	|80.44	|71.65	|70.36	|47.76|
|SCTNet(ours)|	89.22|	93.22|	86.99|	71.21|	70.80|	66.03|



