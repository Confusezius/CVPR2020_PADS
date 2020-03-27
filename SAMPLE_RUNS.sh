"""==================== CUB200 ====================="""
# Provided Weights:
python main.py --source_path $datapath --gpu 0 --seed 3 --n_epochs 300 --beta_lr 0.001 --beta 0.6 --tau 150 --wandb_log --wandb_project_name CVPR2020_PADS --train_val_split 0.85 --wandb_group CUB_V1 --samples_per_class 2 --policy_sample_freq 30 --policy_reward_type 1 --policy_state_metrics recall dists nmi --policy_mode ppo --policy_metric_history 20 --policy_action_values 0.8 1 1.25 --policy_old_update_iter 5 --policy_init_distr uniform_low --policy_n_support 30 --policy_support_limit 0.1 1.4 --loss marginloss



"""==================== CARS196 ====================="""
# Provided Weights:
main.py --dataset cars196 --source_path $datapath --gpu 0 --seed 0 --n_epochs 300 --beta_lr 0.0005 --beta 0.6 --tau 140 --wandb_log --wandb_project_name CVPR2020_PADS --train_val_split 0.9 --wandb_group CAR-T-CHECK-VX4 --samples_per_class 4 --policy_sample_freq 30 --policy_reward_type 2 --policy_state_metrics recall dists nmi --policy_mode ppo --policy_metric_history 20 --policy_action_values 0.8 1 1.25 --policy_old_update_iter 5 --policy_init_distr uniform_low --policy_n_support 30 --policy_support_limit 0.1 1.4 --loss marginloss



"""==================== SOP ====================="""
# Provided Weights:
python main.py --dataset online_products --source_path $datapath --gpu 0 --seed 0 --n_epochs 120 --beta_lr 0.001 --beta 1.2 --tau 80 --wandb_log --wandb_project_name CVPR2020_PADS --train_val_split 0.90 --wandb_group SOP-R50_AV1 --samples_per_class 2 --policy_sample_freq 60 --policy_reward_type 2 --policy_state_metrics recall dists nmi --policy_mode ppo --policy_metric_history 20 --policy_action_values 0.8 1 1.25 --policy_old_update_iter 5 --policy_init_distr uniform_low --policy_n_support 30 --policy_support_limit 0.1 1.4 --loss marginloss --arch resnet50
