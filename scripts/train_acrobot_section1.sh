
# For Acrobot-v1
python src/section1/train_actor_critic_agent.py \
    --env Acrobot-v1 \
    --exp_name acrobot_exp \
    --gamma 0.99 \
    --learning_rate_actor 0.00001 \
    --learning_rate_critic 0.0001 \
    --episodes 1000 \
    --max_steps 500 \
    --early_exit -90 \
    --models_dir assets/section_1/actor_critic/acrobot/models