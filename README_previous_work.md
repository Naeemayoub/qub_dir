# qub_dir
This repo contains following codes:
Sc_gnet_utils.py: it will create the googlenet model
    1.	Inception Layer
    2.	g_net (sc_googlenet model)
    3.	show_final_history (It will sow accuracy and loss of the model and save the fig)
create_training_data.py:
	Two function:
    1.	Func: it the data is in the form of vector (it will return trainx, train y, test x, test y)
    2.	Func: if the data in form of folders (custom folder for manual labeling)
        a.	Input: training dataPath, validation datapath, colortype
        b.	Output: training dataset, validation dataset, labels and in_shape
Training:
	Build the model based on the given argument with addition either ‘rgb’ or ‘gray_images’  
![image](https://user-images.githubusercontent.com/37056548/159931526-b62d3e50-921d-4a56-aa60-a25e36cf4fe8.png)
