<h1 align = "center"> VideoFrameInterpolation_Dance </h1>

<h1 align="center"> VideoFrameInterpolation </h1>

<h1 align="center"> VFI Model </h1>

<h5 align="center"> Capstone Project  (2022-09 ~ ing) </h5>

* This repository is still a work in progress.

`VideoFrameInterpolation_Dance` is the Image interpolation is the process of estimating new pixel values in an image based on the values of surrounding pixels.
We want to perform this image interpolation using computer vision techniques.
We want to focus on interpolation in an image with an object in the center.

We build a new model based on the existing video frame interpolation model "AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation".
`We propose a video frame interpolation model that can be more motion-oriented and object-oriented for dance data, expecting to improve performance and reduce parameters with the new model.`

비디오 프레임 보간은 이전 이후 프레임을 이용하여 가운데 프레임을 추정하는 작업입니다.
우리는 컴퓨터 비전 기술을 이용하여 이를 수행하였습니다.

우리는 기존 "AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation"을 기반으로 하여 새로운 video frame interpolation deeplearning model을 제작하였습니다.

<p float="center">
  <img src="gif/MIRRORED_-Brave-Girls_브레이브-걸스_-Rollin’-안무-거울모드before.gif" alt="Animated gif pacman game" height="240px" width="360px" />
  <img src="gif/MIRRORED_-Brave-Girls_브레이브-걸스_-Rollin’-안무-거울모드after.gif" alt="Animated gif pacman game" height="240px" width="360px" />
</p>

The video file on the left was extracted from YouTube and changed to 12 FPS. We then used our model to interpolate the images to create the right video file at 24 FPS.

As you can see, the right video is much smoother and looks quite natural.

위에 보이는 gif파일은 왼쪽은 유튜브 영상에서 12FPS로 추출한 영상이고 오른쪽은 왼쪽 영상을 이용해 24FPS로 제작한 영상입니다.
오른쪽의 영상이 더욱 부드러운 것을 확인할 수 있습니다.

![stronghold logo](img/interpolation_image1.png)
![stronghold logo](img/interpolation_image2.png)


The photo above shows an image from an earlier point in time, an interpolated image, and an image from a later point in time. The interpolated image is created from the images from the earlier and later viewpoints.

** I'll post the interpolated gif after I get permission from the video's copyright holder. **

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h2> Project Files Description </h2>

<h4>Directories:</h4>
<ul>
  <li><b> dancing folder </b> - Inside the dancing folder, there are folders of models(AdaCoF_ori, AdaCoF_1, AdaCoF_2, AdaCoF_3, AdaCoF_4-2, AdaCoF_5-2) trained with dancing data, each trained in a different way.
  And inside each model folder, we have the model folder, train.py, and trainer.py needed to train the model, and we have the model trained to 50 epochs and 60 epochs.


  <li><b>vimeo folder</b> - The model folder trained with vimeo data is organized like the Vimeo folder.</li>


  <li><b>interpolate_img folder</b> - This is the folder that contains the before and after images for each model.</li>


  <li><b>interpolate_video folder</b> - This is the folder that contains the before and after videos for each model.</li>


  <li><b>output test</b> - This is the folder where you save the results of your image interpolation.</li>


  <li><b>output_youtube_test</b> - Foldup of the interpolation result of Youtube data with a model trained on Dancing data.</li>


  <li><b>output_youtube_test_vimeo</b> - Foldup of the interpolation result of Youtube data with a model trained on Vimeo data.</li>
</ul>



<h4>Executable Files:</h4>
<ul>
  <li><b> interpolate_image.py </b> - A pyfile that runs image interpolation after specifying a particular model.


  <li><b> interpolate_video.py </b> - A pyfile that runs video interpolation after specifying a particular model.</li>


  <li><b>evaluation.py </b> - A pyfile that calculates the PSNR, SSIM score of a model by specifying a specific model and dataset.</li>


  <li><b>MakeTripletset.py </b> - A pyfile that builds images into tripletsets at specific intervals.</li>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


# Model Pipeline
-   The pipeline of our final proposed model is AdaCoF4-2. 
  **I'll post the model pipeline figure after **

The main techniques of the model are described below.
1. Progress in two directions
2. Models that can inject Featuremap differencing information


# Model evaluation
-   Below is a table of results from evaluating the different models we devised on different datasets.

'원본 Adacof' refers to the original Adacof model and the other models are the models we devised.



<h5>We found that our method produced a model with better performance and less computation than the original Adacof model!</h5>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> Paper References</h2>
<ul>
  <li><p>	CVPR 2020 Computer Vision and Pattern Recognition (cs.CV)	arXiv:1907.10244 [cs.CV], 'AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation' [Online].</p>
      <p>Available: https://arxiv.org/abs/1907.10244</p>
  </li>

</ul>

<h2> Code References</h2>
<ul>
  <li><p>	HyeongminLEE
, 'AdaCoF-pytorch' [Online].</p>
      <p>Available: https://github.com/HyeongminLEE/AdaCoF-pytorch</p>
  </li>

</ul>


<h2> Dataset References</h2>
<ul>
  <li><p>Korea Intelligent and Information Society Agency, 'K-pop choreography video' [Online].</p>
      <p>Available: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=52</p>
  </li>
  <li><p>IJCV 2019
Video Enhancement with Task-Oriented Flow, 'Vimeo Triplet dataset (for temporal frame interpolation)'. [Online].</p>
      <p>Available: http://toflow.csail.mit.edu/</p>
  </li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
