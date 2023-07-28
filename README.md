# Accelerating Hair Rendering by Learning High-Order Scattered Radiance
<h6 class="post-title" align="center" style="color:dodgerblue">
<a href="https://aakashkt.github.io/">Aakash KT<sup>1, 2</sup></a>, 
<a href="https://scholar.google.es/citations?user=pXKBhbkAAAAJ&hl=en">Adrian Jarabo<sup>1</sup></a>, 
<a href="http://www.aliagabadal.com/">Carlos Aliaga<sup>1</sup></a>, 
<a href="https://mattchiangvfx.com/">Matt Jen-Yuan Chiang<sup>1</sup></a>, <br> 
<a href="https://www.linkedin.com/in/olivier-maury-7abaa0/">Olivier Maury<sup>1</sup></a>, 
<a href="https://www.linkedin.com/in/christophehery">Christophe Hery<sup>1</sup></a>, 
<a href="https://scholar.google.co.in/citations?user=3HKjt_IAAAAJ&hl=en&oi=ao">P. J. Narayanan<sup>2</sup></a>, 
<a href="https://sites.google.com/view/gjnam">Giljoo Nam<sup>1</sup></a>
</h6> 
<h6 class="post-title" align="center"> <sup>1</sup>Meta Reality Labs Research <br> <sup>2</sup>CVIT, International Institute of Information Technology, Hyderabad (IIIT-H)</h6>

<h4 align="center">Eurographics Symposium on Rendering (EGSR) 2023 <br> Computer Graphics Forum (CGF) Vol. 42, No. 4</h4>

<div align="center">
    <a target="_blank" href="https://aakashkt.github.io/hair_high_order.html">[Project Page]</a>
    <a target="_blank" href="http://cvit.iiit.ac.in/images/ConferencePapers/2023/Hair_Path_Tracing_Acceleration.pdf">[Author's Version PDF]</a>
    <a target="_blank" href="https://diglib.eg.org/handle/10.1111/cgf14895">[Publisher's Version PDF]</a> 
    <a target="_blank" href="https://iiitaphyd-my.sharepoint.com/:b:/g/personal/aakash_kt_research_iiit_ac_in/EWr6HZmMUjlKirzrBvrk1woBCWgrk9Qw6PAJcqrl2TaKaQ?e=6Cgr8k">[Slides]</a>
</div>
<br>

A renderer to path trace hair by using learnt approximations of high order radiance. <br>
The renderer provides control over it's bias (w.r.t. ground truth path tracing) and the speedup. In essence, more speedup results in more bias and vice versa. <br><br>
This bias is however acceptable, depending on the target application. In the best case, we achieve a speedup of approx. 70 % w.r.t. path tracing.

## Cloning & Building
This code is primarily tested with <b>Windows 10</b> and <b>Visual Studio 2022</b>. <br><br>

To clone this repo, run the following (note the ```--recursive``` flag):
```
git clone --recursive https://github.com/facebookresearch/HairMSNN
```
<br>
Next, create a build directory and run CMake:
```
cd PATH_TO_CLONED_REPO
mkdir build
cd build
cmake ..
```
<br>

Open the resulting solution in Visual Studio and build in ```Release``` configuration. It should build the following:
```
render_path_tracing -> Path tracing
render_nrc -> Neural Radiance Caching
render_hair_msnn -> Our renderer
```

## Example Scenes
This repo included two example scenes of two different hair styles: Straight & Curly. Both styles and the head model are taken from <a href="http://www.cemyuksel.com/research/hairmodels/">Cem Yuksel's webpage</a>. <br>
We have a custom scene description file, which can be found in ```scenes/straight/config.json``` or ```scenes/curly/config.json```. <br>
<b>Please note, you will have to edit the paths in the scene file to use them</b>. For example, to use ```scenes/curly/config.json```, modify the highlighted lines:
```json
{

    "hair": {
        "alpha": 2.0,
        "beta_m": 0.3,
        "beta_n": 0.3,
        "type": 1,
        "sigma_a": [
            0.06,
            0.1,
            0.2
        ],
        "geometry": *"YOUR_REPOSITORY_LOCATION/scenes/curly/wCurly.hair"
    },
    "integrator": {
        "ENV_PDF": true,
        "MIS": true,
        "height": 1024,
        "image_output": *"YOUR_REPOSITORY_LOCATION/scenes/straight/render.png",
        "path_v1": 1,
        "path_v2": 40,
        "spp": 500,
        "stats_output": *"YOUR_REPOSITORY_LOCATION/scenes/straight/stats.json",
        "width": 1024
    },
    "lights": {
        "environment": {
            "exr": *"YOUR_REPOSITORY_LOCATION/scenes/envmaps/christmas_photo_studio_07_4k.exr",
            "scale": 1.0
        },
        "directional": [
            {
                "from": [3, 3, 3],
                "emit": [1, 1, 1]
            }
        ]
    },
    "tcnn": {
        "config": *"YOUR_REPOSITORY_LOCATION/scenes/curly/tcnn_hairmsnn.json",
        "init_train": true
    },
    "surface": {
        "geometry": *"YOUR_REPOSITORY_LOCATION/scenes/curly/head.obj"
    }
}
```

## Usage
The general syntax for interactive rendering is:
```
./Release/render_[RENDERER].exe [SCENE_FILE_PATH] [BETA]
```
where ```[RENDERER]``` can be one of ```path_tracing```, ```nrc``` or ```hair_msnn```. <br>
```[SCENE_FILE_PATH]``` is the path to the scene file. <br>
The integer argument ```[BETA]``` is processed only by our renderer (```hair_msnn```) and controls the bias/speedup. For small values (0, 1, 2 ..), the speedup is considerable and bias exists. For higher values (10, 11 ..), the run-time & bias approaches path tracing. Refer to the paper for details.
<br><br>

Run path tracing on curly hair:
```
./Release/render_path_tracing.exe YOUR_REPOSITORY_LOCATION/scenes/curly/config.json
```
<br>

Run NRC on curly hair:
```
./Release/render_nrc.exe YOUR_REPOSITORY_LOCATION/scenes/curly/config.json
```
<br>

Run our renderer on curly hair, with maximum speedup & bias (BETA=1):
```
./Release/render_hair_msnn.exe YOUR_REPOSITORY_LOCATION/scenes/curly/config.json 1
```

The ```Save EXR``` and ```Save PNG``` buttons in the UI will save the current frame in ```image_output``` defined above. <br>
There are quite a few other parameters in the UI that you can play with, which should all make intuitive sense.

## Requirements
- Tested on a Windows 10 machine with RTX 3090 & Visual Studio 2022
- A recent NVIDIA GPU, with RTX cores
- CMake v3.18
- A C++14 capable compiler
- NVIDIA Cuda 12.1
- NVIDIA OptiX 7.4
- NVIDIA driver 531

## License
HairMSNN is MIT licensed, as found in the LICENSE file.

## Citation
If you find this repository useful in your own work, please consider citing the paper.
```
@article{10.1111:cgf.14895,
  journal = {Computer Graphics Forum},
  title = {{Accelerating Hair Rendering by Learning High-Order Scattered Radiance}},
  author = {KT, Aakash and Jarabo, Adrian and Aliaga, Carlos and Chiang, Matt Jen-Yuan and Maury, Olivier and Hery, Christophe and Narayanan, P. J. and Nam, Giljoo},
  year = {2023},
  publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
  issn = {1467-8659},
  doi = {10.1111/cgf.14895}
}
```
