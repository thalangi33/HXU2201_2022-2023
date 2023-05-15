## Installation
1. Follow the installation instructions from mmaction2 https://github.com/open-mmlab/mmaction2
2. Add the files to mmaction2 repo directory to run the python programs
3. Older version of mmaction2 was used

## Frame selection strategies
top_k: Select frames with the lowest similarity scores.

3_sections: Not all sections are equal. Some sections were more important. Videos will be split equally into 3 sections. More important sections will have
higher number of frames. Importance is determined by standard deviation of image similarity scores from each section.

uniform: Every section is equal. The number of frames is split evenly between sections. Lowest similarity score frames of each section will be selected for the analysis.

alt_uniform: Split the video into 12 sections. First frame of each section and first frame of its adjacent section will be compared for similarity score. Sections with lower scores are deemed more important, thus will receive more frames. Frames will be selected randomly from each section. Only 12 image similarity scores are needed.

## Image similarity scores
CV2 and SSIM were used to generate image similarity scores

## MGSampler
To learn how to use MGSampler, please refer to https://github.com/lissa2077/MGSampler"# HXU2201_2022-2023" 
