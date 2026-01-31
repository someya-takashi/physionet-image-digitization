mkdir /root/.kaggle
mv kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

kaggle datasets download -d hengck23/hengck23-demo-submit-physionet
unzip hengck23-demo-submit-physionet.zip -d input/hengck23-demo-submit-physionet
rm hengck23-demo-submit-physionet.zip
kaggle datasets download -d takashisomeya/physionet-ecg-image-digitization-meta
unzip physionet-ecg-image-digitization-meta.zip -d input/physionet-ecg-image-digitization
rm physionet-ecg-image-digitization-meta.zip

mkdir data/rectified
kaggle datasets download -d takashisomeya/physionet-rectified-images-fold0
unzip physionet-rectified-images-fold0.zip -d data/rectified
rm physionet-rectified-images-fold0.zip

kaggle datasets download -d takashisomeya/physionet-rectified-images-fold1
unzip physionet-rectified-images-fold1.zip -d data/rectified
rm physionet-rectified-images-fold1.zip

kaggle datasets download -d takashisomeya/physionet-rectified-images-fold2
unzip physionet-rectified-images-fold2.zip -d data/rectified
rm physionet-rectified-images-fold2.zip

kaggle datasets download -d takashisomeya/physionet-rectified-images-fold3
unzip physionet-rectified-images-fold3.zip -d data/rectified
rm physionet-rectified-images-fold3.zip

kaggle datasets download -d takashisomeya/physionet-rectified-images-fold4
unzip physionet-rectified-images-fold4.zip -d data/rectified
rm physionet-rectified-images-fold4.zip

mv data/rectified/rectified_fold0/* data/rectified/
mv data/rectified/rectified_fold1/* data/rectified/
mv data/rectified/rectified_fold2/* data/rectified/
mv data/rectified/rectified_fold3/* data/rectified/
mv data/rectified/rectified_fold4/* data/rectified/
