python "./training/train_corrector.py"^
 --dataset=multi_faces --epochs=10 --batch_size=1 --lr=0.0005 --model=cnet --diff_features_only
@REM  --stage_start=5 --stage_end=5^
@REM  --train_size=2000 --test_size=200 --val_size=200 --h_min=3 --h_max=11 --n_min=3 --n_max=11^
@REM  --predictor_checkpoint="hpc/results/predictor_single_cube_cnet_mish_radam_sd/2020-12-06_164033/predictor_checkpoints.t7"^
@REM  --detector_checkpoint="hpc/results/detector_single_cube_cnet_mish_radam_sd/2020-12-06_103922/detector_checkpoints.t7"
@REM  --corrector_checkpoint="training/checkpoints/corrector_01_single_bunny_cnet_mish_radam_sd/2020-12-04_205947/corrector_checkpoints.t7"^