# 3D_Movie_Generation

## Steps to run the code

1. Download the Holopix50k dataset from [here](https://leiainc.github.io/holopix50k/) and place it in the dataset folder

2. Create a virtual environment and install the requirements using the following commmands
    ```
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt```
    
3. To train the model, run
    ```python3 train_stereo_pairs.py```

4. To generate a 3D video from a given 2D video, run
    <br>``` python3 2D_to_3D_movie_converter.py <path_to_video.mp4>```
    

<br><br>Use your 3D glasses and enjoy the movie!
