
# CF_DIST_ID=[Provided cloudfront distribution id]


mkdir -p sample-data
cd sample-data
wget https://$CF_DIST_ID.cloudfront.net/endo-vit/sample_segmentation.tar.gz
wget https://$CF_DIST_ID.cloudfront.net/endo-vit/sample_validation.tar.gz

tar zxvf sample_segmentation.tar.gz
tar zxvf sample_validation.tar.gz

cd ..
mkdir -p pt-models
cd pt-models
wget https://$CF_DIST_ID.cloudfront.net/endo-vit/mae_pretrain_vit_base_full.pth

