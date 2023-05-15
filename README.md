## Installation
1. Follow the installation instructions from [mmaction2](https://github.com/open-mmlab/mmaction2)
2. Add the files to mmaction2 repo directory to run the python programs
3. Older version of mmaction2 was used

## Frame selection strategies
1. [top_k.py](https://github.com/thalangi33/HXU2201_2022-2023/blob/main/top_k.py): Select frames with the lowest similarity scores.
![image](https://github.com/thalangi33/HXU2201_2022-2023/assets/99813905/f3f284dd-bb6e-4d68-aead-64f973bcdcbd)


2. [3_sections.py](https://github.com/thalangi33/HXU2201_2022-2023/blob/main/3_sections.py): Not all sections are equal. Some sections were more important. Videos will be split equally into 3 sections. More important sections will have higher number of frames. Importance is determined by standard deviation of image similarity scores from each section.
![image](https://github.com/thalangi33/HXU2201_2022-2023/assets/99813905/c2b32753-9b6e-4b55-a397-94d4ce1e5394)

3. [uniform.py](https://github.com/thalangi33/HXU2201_2022-2023/blob/main/uniform.py): Every section is equal. The number of frames is split evenly between sections. Lowest similarity score frames of each section will be selected for the analysis.
![image](https://github.com/thalangi33/HXU2201_2022-2023/assets/99813905/80631ab0-7256-47eb-a02f-bc84463d4cf7)

4. [alt_uniform.py](https://github.com/thalangi33/HXU2201_2022-2023/blob/main/alt_uniform.py): Split the video into 12 sections. First frame of each section and first frame of its adjacent section will be compared for similarity score. Sections with lower scores are deemed more important, thus will receive more frames. Frames will be selected randomly from each section. Only 12 image similarity scores are needed.
![image](https://github.com/thalangi33/HXU2201_2022-2023/assets/99813905/5e365bcc-7cf2-4be2-8f05-d5f25785e272)

## Image similarity scores
[CV2](https://github.com/thalangi33/HXU2201_2022-2023/blob/main/cv2_generate.py) and [SSIM](https://github.com/thalangi33/HXU2201_2022-2023/blob/main/ssim_generate.py) were used to generate image similarity scores

## MGSampler
To learn how to use MGSampler, please refer to lissa2077's [MGSampler](https://github.com/lissa2077/MGSampler)
