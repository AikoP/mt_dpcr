@REM python "./utils/generator.py" --train_size=1 --test_size=0 --val_size=1 --dataset=multi_abc --h_min=10 --h_max=20 --n_min=5 --n_max=15

@REM python "./utils/generator.py" --train_size=200 --test_size=20 --val_size=20 --dataset=single_cube --h_min=3 --h_max=11 --n_min=3 --n_max=9

python "./utils/generator.py" --type=corrector --train_size=500 --test_size=1 --val_size=250 --dataset=multi_faces --h_min=3 --h_max=11 --n_min=3 --n_max=11^
 --predictor_checkpoint="hpc/results/predictor_multi_faces_cnet_mish_radam_sd/2020-12-15_224134/predictor_checkpoints.t7"^
 --detector_checkpoint="hpc/results/detector_multi_faces_cnet_mish_radam_sd/2020-12-09_074025/detector_checkpoints.t7"^
 --max_pipeline_iterations=5
 @REM  --corrector_checkpoint="training/checkpoints/corrector_01_single_bunny_cnet_mish_radam_sd/2020-12-04_205947/corrector_checkpoints.t7"^

@REM python "./utils/generator.py" --type=corrector --train_size=5000 --test_size=1 --val_size=500 --dataset=single_bunny --h_min=3 --h_max=11 --n_min=3 --n_max=9^
@REM  --predictor_checkpoint="hpc/results/predictor_single_bunny_cnet_mish_radam_sd/2020-12-06_104726/predictor_checkpoints.t7"^
@REM  --detector_checkpoint="hpc/results/detector_single_bunny_cnet_mish_radam_sd/2020-12-08_110946/detector_checkpoints.t7"^
@REM  --max_pipeline_iterations=5

@REM python "./utils/generator.py" --type=corrector --train_size=500 --test_size=1 --val_size=50 --dataset=multi_simple_shapes --h_min=3 --h_max=6 --n_min=3 --n_max=6^
@REM  --predictor_checkpoint="hpc/results/predictor_multi_simple_shapes_cnet_mish_radam_sd/2020-12-19_062318/predictor_checkpoints.t7"^
@REM  --detector_checkpoint="hpc/results/detector_multi_simple_shapes_cnet_mish_radam_sd/2020-12-15_224339/detector_checkpoints.t7"^
@REM  --max_pipeline_iterations=5