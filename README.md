If you would like to train the model on CelebA, then you need to manually download celeba by running:
```
mkdir data
cd data
curl -O https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
unzip celeba
mkdir celeba 
mv img_align_celeba celeba
rm celeba.zip
```