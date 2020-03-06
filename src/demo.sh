# EDSR baseline model (x2) + JPEG augmentation
#python3 main.py --model EDSR --scale 4 --save edsr_x4 --reset --data_test Set5+Set14+B100+Urban100+DIV2K --n_GPUs 1 --epochs 300 --dir_data ../../datasetx4 --reset
#python3 main.py --model EDSR --scale 4 --save edsr_x4_spl --reset --data_test Set5+Set14+B100+Urban100+DIV2K --n_GPUs 1 --epochs 300 --dir_data ../../datasetx4 --reset
# test
#python main.py --model EDSR --scale 2 --test_only --dir_data ../../dataset/testx2 --n_GPUs 1 --data_test Set5+Set14+B100+Urban100 --pre_train ../experiment/edsr_baseline_x2_L1/model/model_best.pt

#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python3 main.py --template EDSR_paper --scale 2 --save edsr_x2_spl_1011 --n_GPUs 1 --patch_size 96 --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2 --resume -1
#python3 main.py --template EDSR_paper --save edsr_x4_spl_test --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --test_only --pre_train ../experiment-train/edsr_x4_spl_1013/model/model_best.pt --reset
#python3 main.py --template EDSR_paper --save edsr_x4_spl_test --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --test_only --pre_train ../experiment-train/edsr_x4_spl_1013/model/model_latest.pt --reset
# EDSR in the paper (x3) - from EDSR (x2)
#python3 main.py --template EDSR_paper --scale 3 --save edsr_spl_x3_1012 --reset --n_GPUs 1 --patch_size 144 --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx3 --pre_train /media/E/linwei/SISR/EDSR-PyTorch-SPL/experiment/edsr_x2_spl_1011/model/model_best.pt
#python main.py --model EDSR --scale 3 --save edsr_x3_spl_1012 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]
#python3 main.py --template EDSR_paper --scale 4 --save edsr_x4_spl_1013 --n_GPUs 1 --patch_size 192 --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --resume -1
#python3 main.py  --template SRRESNET --data_test B100 --scale 4 --pre_train ../experiment-train/SRRESNETx4_SPL_1011/model/model_latest.pt --test_only --save_results --dir_data ../../datasetx4 --n_GPUs 1 --save test_SRResNet_spl_b100
# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

#VDSR
#python3 main.py --template VDSR --save VDSR_x2_spl_1009 --scale 2 --reset --patch_size 96 --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2 --n_GPUs 1
#python3 main.py --template VDSR --save VDSR_x3_spl_1009 --scale 3 --reset --patch_size 144 --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx3 --n_GPUs 1
#python3 main.py --template VDSR --save VDSR_x4_spl_1009 --scale 4 --reset --patch_size 192 --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1

#python3 main.py --template VDSR --save vdsr_x2_spl_test --scale 2 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2 --n_GPUs 1 --test_only --pre_train ../experiment-train/VDSR_x2_spl_1009/model/model_best.pt --reset
#python3 main.py --template VDSR --save vdsr_x3_spl_test --scale 3 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx3 --n_GPUs 1 --test_only --pre_train ../experiment-train/VDSR_x3_spl_1009/model/model_best.pt --reset
#python3 main.py --template VDSR --save vdsr_x4_spl_test --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --test_only --pre_train ../experiment-train/VDSR_x4_spl_1009/model/model_latest.pt --reset


# LapSRN
#python3 main.py --template LapSRN --save LapSRN_x2_spl_1010 --scale 2 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2 --n_GPUs 1 --patch_size 128
#python3 main.py --template LapSRN --save LapSRN_x3_spl_1010 --scale 3 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx3 --n_GPUs 1 --patch_size 127
#python3 main.py --template LapSRN --save LapSRN_x4_spl_1010 --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --patch_size 128

#python3 main.py --template LapSRN --save lapsrn_x2_spl_test --scale 2 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2 --n_GPUs 1 --test_only --pre_train ../experiment-train/LapSRN_x2_spl_1010/model/model_best.pt --reset
#python3 main.py --template LapSRN --save lapsrn_x4_spl_test --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --test_only --pre_train ../experiment-train/LapSRN_x4_spl_1010/model/model_best.pt --reset

# DRRN
#python3 main.py --template DRRN --save DRRN_x2_spl_1010 --scale 2 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2 --n_GPUs 2
#python3 main.py --template DRRN --save DRRN_x3_spl_1010 --scale 3 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx3 --n_GPUs 2
#python3 main.py --template DRRN --save DRRN_x4_spl_1010 --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 2

#python3 main.py --template DRRN --save drrn_x4_spl_test --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --test_only --pre_train ../experiment-train/DRRN_x4_spl_1010/model/model_best.pt --reset

# SRCNN
#python3 main.py --template SRCNN --save SRCNN_x2_spl_1010 --scale 2 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2 --n_GPUs 1
#python3 main.py --template SRCNN --save SRCNN_x3_spl_1010 --scale 3 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx3 --n_GPUs 1
#python3 main.py --template SRCNN --save SRCNN_x4_spl_1019 --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1

#python3 main.py --template SRCNN --save SRCNN_x4_spl_test --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --test_only --pre_train ../experiment-train/SRCNN_x4_spl_1010/model/model_best.pt --reset

# SRResNet
#python3 main.py --template SRRESNET --scale 4 --save SRRESNETx4_1012 --reset --save_models --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1

#python3 main.py --template SRRESNET --save SRRESNET_x4_baseline_test --scale 4 --reset --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx4 --n_GPUs 1 --test_only --pre_train ../experiment-train/SRRESNETx4_1012/model/model_best.pt --reset

# MDSR
# python3 main.py --template MDSR --scale 2+4 --n_resblocks 80 --save MDSR_spl_1012 --reset --save_models --n_GPUs 1 --data_test Set5+Set14+B100+Urban100 --dir_data ../../datasetx2
