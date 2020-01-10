This simulation code was tested using the following: 

System: Ubuntu 16.04 or later (64-bit)

Python: Python 3.6.8

pip: pip 19.3.1

virtualenv: 16.6.2




Step 1: unzip model.tar.gz and change directory to direcotry the newly-created folder named model 

Step 2: Create virtual environment by using the following commands
 
./install_essentials.sh  
virtualenv -p python3 venv
source venv/bin/activate 

Step 3: use the following command to install remaining dependencies  
 
pip3 install -r requirements.txt

Step 4: Use the following command to execute the model. The prediction accuracy will be reported in a file named accuracy.csv. (To change any variable or parameter, edit the parameter block in the source file "main.py".

python3 main.py   



