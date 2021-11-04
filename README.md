# SMPL-LSTM
Predicting SMPL using LSTM

## How to run
### Data extraction and visualization  
1. download model.pkl file from Drive or smplModels repo and place it in root directory
2. Run *preprocess_pedx.py* for on srcDir i.e directory contains images, it will rename the images suitable for vibe demo
3. copy demo_modified.py into vibe root directory and run the following command
   `python demo_modified.py --images_dir /home/anil/Documents/VIBE/srcFiles --output_folder output `  
   it will generate *vibe_output.pkl* file
4. run *getSMPLFromVibe.py*. It will generate out.json - which contains the SMPL pose and camera parameters
   1. Specify location of *vibe_output.pkl* file in code
5. run *viewMeshes.py* to render the generated meshes
   1. Specify source image directory, out.json locations

### Running LSTM Model
1. Preprocess extracted data i.e out.json using *preprocess.py*
2. run *smplLstm.py* 

### To visualize tensorboard
` tensorboard --logdir logs`


### Running VIBE
VIBE by default take video file as input and  generate/renders/overlays SMPL models on images and outputs a video. 
Since we are using images , I added those to *demo_modified.py*. So Copy it to VIBE root directory and run following command
by specifying source image dir and destination image dir.
Rendered output images will be stored in destination image dir 

**MIN_NUM_FRAMES** : Is used to specify minimum number of frames a person has to present in video to be processed by VIBE. I'm currently using it as 6

### Preprocessing
I'm currently using a lookback window to make data samples. for example for lookback window of length l, if person p is in
n number of frames. then number of data points for person p will be

ndp = ( n - l ) * ( 72 ) 
 
72 -> SMPL pose parameters  
3 -> translation parameters

I still need to find out how to calculate translation parameters

```
>>> x = [1,2,3,4,5,6]
>>> out = []
>>> for i in range(3,len(x)+1) :
...     out.append(x[i-3:i])
... 
>>> out
[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]

```

### Pipeline 
*pipeline.py* is used to process all images in input dir using VIBE. Curretly it devides images in small batches and process it. Later those are combined in JSON.

One of the drawback of this approach is we can't cross reference parameters with images.


### visualizing SMPL data 
VIBE generates and renders Vertices, since we are currently intrested in SMPL parameters I need to generate/pose SMPL model 
based on SMPL pose parameters. I found the [Numpy SMPL](https://github.com/CalciferZh/SMPL) implementation , which generates
SMPL mesh / vertices using pose parameters.

So I will extract the pose parameters from VIBE and generate SMPL mesh from extracted pose parameters and display using pyrender.
renderer borrowed from VIBE implementation.

Before running the visualization get/generate *model.pkl* using [SMPL](https://github.com/CalciferZh/SMPL). I hosted modified pkl file in [my private repo](https://github.com/anilkunchalaece/smplModels) and Drive [*TU Dublin -> SMPL*]

### Evaluation 
Evaluation performed by rendering SMPL vertex using SMPL parameters. Then vertex to vertex error, MPJPE, MPJAE will be calculated.

SMPL rendering also carried out in steps due to limited computational power.


## References
- [https://github.com/ikvision/ikvision.github.io/blob/master/README.md#5-predicted-camera-parameters](https://github.com/ikvision/ikvision.github.io/blob/master/README.md#5-predicted-camera-parameters)

## Problem with GPU
When running on pytorch-gpu, smpl_torch_batch is giving very bad (trust me , I wasted a day on this ) results. So I'm sticking with running using CPU when running smpl_torch_batch. For Training and testing I'm using GPU

## Problem with Large BEHAVE Data
instead of loading large file in to ram, I split the large file into multiple files each one consisting data about per person

## Datasets 
1. Pedx.io Dataset - [http://pedx.io/](http://pedx.io/)
2. JAAD Dataset - [http://data.nvision2.eecs.yorku.ca/JAAD_dataset/](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
3. TUD Pedestrains -[https://www.pgram.com/dataset/tud-pedestrians/](https://www.pgram.com/dataset/tud-pedestrians/)
4. **Titan Dataset** - [https://usa.honda-ri.com/titan](https://usa.honda-ri.com/titan) - looks super useful
5. CityPersons - [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)