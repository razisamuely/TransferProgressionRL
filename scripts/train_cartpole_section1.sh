python src/section1/train_actor_critic_agent.py \
    --env CartPole-v1 \
    --exp_name cartpole_exp \
    --gamma 0.99 \
    --learning_rate_actor 0.00001 \
    --learning_rate_critic  0.0001 \
    --episodes 3000 \
    --max_steps 1000 \
    --early_exit 475 \
    --models_dir assets/section_1/actor_critic/cart_pole/models
