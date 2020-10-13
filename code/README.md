# A Samplable Multimodal Observation Model for Global Localization and Kidnapping

![](figs/DSOM.png)

<center><b>Stucture of Deep Multimodal Observation Model</b></center>
## Requirements

To install requirements:

```setup
conda create --name <your_env_name> --file requirements.txt
```

## Training

To generate synthetic datasets for training: 

```generate synthetic dataset
cd Data_processor
python <dataset>
```

\<dataset\> can be Bicocca_rc_dataset.py, synthetic_dataset.py, or UPO_rc_dataset. 



To train the model in the paper, run this command:

```train
python Main.py --operation train --dataset <dataset> --subdataset <subdataset>
```

## Evaluation

You can download the evaluation sequences here and put them in directory DMOM/Dataset/

* [Evaluation Datasets](https://drive.google.com/drive/folders/1yhriJFvkGKeLnF3LBo5Yt-eOJSNQ1y45?usp=sharing). 

To evaluate DMOM and Adaptive Mixture MCL:

```eval
python Main.py --operation test_MCL --dataset <dataset> --subdataset <subdataset> --update_mode Adaptive_Mixture_MCL --w_cut 0.6 --particle_num 200 --load_model <path_to_model>
```

## Pre-trained Models

You can download pretrained models here and put them in directory DMOM/Trained_models/

- [Pretrained Models](https://drive.google.com/drive/folders/1yhriJFvkGKeLnF3LBo5Yt-eOJSNQ1y45?usp=sharing). 

## Results

Our model achieves the following global localization and kidnapping performance on our synthetic dataset.

![](figs/gl_error.jpg)



![](figs/kn_error.jpg)



![](figs/gl_conv_results.jpg)



![](figs/kn_conv_results.jpg)



Our model achieves the following global localization and kidnapping performance on the UPO dataset collected in a real palace.

![](figs/results_UPO.png)