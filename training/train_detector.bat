python "./training/train_detector.py"^
 --train_data=single_bunny^
 --model=cnet --model_cap=normal^
 --epochs=15 --batch_size=3 --lr=0.0001^
 --safe_descent --dynamic_lr --diff_features_only^
 --noise=0.1 --scale_min=0.9 --scale_max=1.1