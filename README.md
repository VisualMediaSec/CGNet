### CGNet Dataset

Download three CG image detection datasets:
- `DSTok`: 4850 natural images and 4850 CG images from the [DSTOK](https://www.ic.unicamp.br/~rocha/pub/communications).
- `He`: 6800 natural images and 6800 CG images, this dataset can be requested from [He et al.]（https://ieeexplore.ieee.org/document/8410682）.
- `Rahmouni`: 1800 natural images and 1800 CG images, refer to[NicoRahm/CGvsPhoto](https://github.com/NicoRahm/CGvsPhoto)

Note that，each image in above datasets have been cut into 20 image blocks with a fixed resolution of 224*224, and you need reference the detailed description in Section 4.2 (Experimental Settings) of our paper. 


## Citation
```
@Article{yao_cgnet_2022,
	author="Ye Yao
	and Zhuxi Zhang
	and Xuan Ni
	and Zhangyi Shen
	and Linqiang Chen
	and Dawen Xu",
	title="CGNet: Detecting computer-generated images based on transfer learning with attention module",
	journal="Signal Processing: Image Communication",
	year="2022",
	pages="116692",
	issn="0923-5965"}"
}
```
