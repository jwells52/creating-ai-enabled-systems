# mkdir /root/.kaggle
# cp /content/drive/MyDrive/kaggle.json /root/.kaggle/kaggle.json 
# chmod 600 /root/.kaggle/kaggle.json 

mkdir $1
cp $2 $1/kaggle.json
chmod 600 $1/kaggle.json
kaggle competitions download -c humpback-whale-identification
unzip humpback-whale-identification.zip