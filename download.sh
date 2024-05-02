rm /content/EuroSAT.zip
rm -r /content/2750
rm -r /content/dataset

wget https://madm.dfki.de/files/sentinel/EuroSAT.zip --no-check-certificate || curl -O https://madm.dfki.de/files/sentinel/EuroSAT.zip

unzip /content/EuroSAT.zip
mkdir /content/dataset

mkdir /content/dataset/validation
mv /content/2750 EuroSAT
mv /content/EuroSAT /content/dataset/
cd /content/dataset
mv EuroSAT training
cd ..

python /content/QML-tutorial/scripts/split_dataset.py

rm /content/EuroSAT.zip