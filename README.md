## DSMNet

### About
This repo contains the code and files necessary to reproduce the results published in our [paper](https://arxiv.org/pdf/2011.10697.pdf) 'Height Prediction and Refinement from Aerial Images with Semantic and Geometric Guidance'. Our method relies on a two stage pipeline : First, a multi-task network is used to predict the height, semantic labels and surface normals of an input RGB aerial image. Next, we use a denoising autoencoder to refine our height prediction in order to produce higher quality height maps. Training and testing is conducted on two publicly available datasets : The ISPRS Vaihingen and the IEEE DFC2018.

![](/images/fullnet.png)



## Network
### Prerequisites

* Python 3.5
* Tensorflow 2.1 (with Cuda 10.0)
* Numpy 1.18.4

You can also refer to the `requirements.txt` file for a complete list of dependencies.

### Datasets
The Vaihingen dataset can be downloaded from the [ISPRS 2D Semantic Labeling Contest page](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx?utm_source=chatgpt.com). Please follow the instructions on the website to request access to the data.
When unzipping the datasets and checkpoints, ensure the following folder structure is maintained:

```
/
├── datasets
│   ├── DFC2018
│   │   ├── RGB
│   │   ├── SEM
│   │   ├── DSM
│   │   └── DEM
│   └── Vaihingen
│       ├── RGB
│       ├── SEM
│       └── NDSM
└── checkpoints
```

### Training
To train the Multi-Task Learning (MTL) prediction network, run:  
```bash
python train_mtl.py [dataset]
```
For example, to train on the DFC2018 dataset:
```bash
python train_mtl.py DFC2018
```

![](/images/mtl_output.png)

Before training the refinement network, ensure you have a trained MTL checkpoint. Then, start training the refinement network with:
```bash
python train_ec.py [dataset]
```
For example, to train on the Vaihingen dataset:
```bash
python train_ec.py Vaihingen
```

![](/images/refinement_output.png)

### Testing 

To evaluate the height prediction and refinement networks, use the `test_dsm.py` script:

```bash
python test_dsm.py [dataset] [refinement_flag]
```

- `[dataset]`: Name of the dataset to test on (`DFC2018` or `Vaihingen`).
- `[refinement_flag]`: Set to `True` to test both prediction and refinement networks, or `False` to test only the prediction network.

**Examples:**

- Test both prediction and refinement on the DFC2018 dataset:
      
      ```bash
      python test_dsm.py DFC2018 True
      ```
- Test only the prediction network on the Vaihingen dataset:
      
      ```bash
      python test_dsm.py Vaihingen False
      ```

The output files will be saved in the `/output` directory.


### Citation
If you find our work useful in your research, please consider citing our [paper](https://arxiv.org/pdf/2011.10697.pdf):

```
@misc{mahdi2020height,
      title={Height Prediction and Refinement from Aerial Images with Semantic and Geometric Guidance}, 
      author={Elhousni Mahdi and Huang Xinming and Zhang Ziming},
      year={2020},
      eprint={2011.10697},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```



