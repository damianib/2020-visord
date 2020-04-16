# Computer Vision Project 2020

Welcome to the repository of our Computer Vision project !

The project was initiated as an academic project at [CentraleSupélec](https://www.centralesupelec.fr/)
during the Computer Vision course.

The goal of the project is to use Computer Vision methods to detect the heartbeat of someone using a video of their face.
The heartbeat is detected simply by using the color variation of the face, which becomes redder once per heartbeat.

The video is processed in 3 steps : spatial processing, temporal processing (using a bandpass filter) and the final merge operation.
**More details about this in our report !**

After the video is processed, you obtain the value of the bpm and a video where the color variation of the face (variation 
of red intensity) are exagerated, to make the heartbeat visible.

This project is mainly based on a thesis about Video Magnification from the MIT : http://people.csail.mit.edu/mrub/vidmag/.

## Run the project

### Prerequisites
Before being able to run this project, you need to fulfill the following
requirements:

* [Python 3.8](https://www.python.org/)

The following Python module are required :

* [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/)
* [OpenCV](https://opencv.org/)

### Detect a hearbeat - oh god it's alive !
The project does not have any CLI or GUI.

To use it simply edit the parameters in the main.py file :
* **source_path** is the path of the source video file in which you want to highligh the heartbeat of someone
* **out_path** is the path of the output video
* **downsample_level** is the number of time you will perform the blur and downsampling operations for the spatial processing
* **lowcut**, **highcut**, **fs** and **order** are for the bandpass filter
* **alpha**, **chrome_attenuation** and **distance_threshold** are for the merge operation at the end

You should probably just change the **source_path** and the **out_path**, but maybe changing the other parameters will 
help you get a better result _for your specific case_.

And then run the main.py using : 

```shell
python main.py
```

For better result, you should use a video which shows only the face of one people, standing as still as possible.
 
## Built With
The whole project is written in [Python 3](https://www.python.org/).

## Contributing
This project does not accept public contributions.

## Authors
* [Hackatosh](https://github.com/Hackatosh)
* [damianib](https://github.com/damianib)
* [Zizol](https://github.com/Zizol)

## License
This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Acknowledgments
* Thanks to CentraleSupélec.
* Thanks to the people at the MIT who made the thesis and the original implementation in C.
