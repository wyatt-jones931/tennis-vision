# tennis-vision
Amateur tennis lacks accessible match analytics tools. While some exist today, they are often costly and have usage rate limitations for the user, limiting their overall usefulness. This repo is an attempt to address this gap in tool that uses current computer vision models to process and produce meaningful analytics from unprofessional tennis footage.

## How to Use This Repo
The first step is to go out and play tennis! The footage I used when testing this model was shot from a GoPro Hero 11, but just about any personal camera should work, including your cellphone. The camera should be able to see all parts of the court and be behind the player you wish to track, but can handle lower/off angles as long as the corners of the court are distinct enough.

Once you have footage and you have a copy of this repo, create a Python environment with opencv-python, numpy, pandas, ultralytics, matplotlib, and scipy installed.  You will then need to create several folders with the following structure:

.  
├── data/  
│   ├── video_files/  
│   ├── frames/  
│   ├── output_frames/  
│   └── output_video_files/  
├── homography/  
└── models/  

Then, move a .MP4 file of your video into the video_files/ folder as video_name.MP4.

Once the project is setup, then run ```python ingest.py video_name --fps 5```. *If you wish to save a csv of the video analytics alongside the .db file, add the flag ```--save_csv```. If the same video name already exists in the database, but you wish to overwrite it, add the flag ```--overwrite```.*

```ingest.py``` will first initialize the database schema from tennis.db

## Pre-Processing Video
<p align="center"><img src="https://github.com/wyatt-jones931/tennis-vision/blob/main/Examples/BaselineElevatedViewPre.gif"></p>

## Post-Processing Video
<p align="center"><img src="https://github.com/wyatt-jones931/tennis-vision/blob/main/Examples/BaselineElevatedViewPost.gif"></p>

## Player Position Heatmap
<p align="center"><img src="https://github.com/wyatt-jones931/tennis-vision/blob/main/Examples/TennisVisionHeatMap.png" width="500"></p>

## Future Work
