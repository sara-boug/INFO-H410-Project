# INFO-H410-Project
## Generate the folder structure 
Ensure the following structure of the data folder, optionally run bash file to generate the folders **generate_data_folder.sh**
### Folder structure
/INFO-H410-Project <br />
&ensp;  /data <br />
&ensp;    /dataset <br />
 &ensp;&ensp;&ensp;      /images <br />
 &ensp;&ensp;&ensp;      /masks <br />
 &ensp;   /generated_models <br />
 &ensp;   /loaded_data <br />
  &ensp;  /preprocessed_data <br />
  &ensp;&ensp;&ensp;    /all <br />
  &ensp;&ensp;&ensp;    /test_set <br />
  &ensp;&ensp;&ensp;    /training_set <br />
  &ensp;&ensp;&ensp;    /validation_set <br />

### Generate the training data 
run : python main.py --preprocess
### Train the model 
run : python main.p --preprocess
### Show the training evolution 
run : python main.py --show-evol
### Perform a prediction 
run: python main.py --test 
