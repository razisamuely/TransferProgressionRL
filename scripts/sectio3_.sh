echo "00001 00001 00001 00001"
#python src/section3/main.py     --env MountainCarContinuous-v0     --exp_name mountaincar_exp     --gamma 0.99     --learning_rate_actor 0.00001     --learning_rate_critic 0.0001     --episodes 2000     --max_steps 2000         --models_dir assets/section_3_alt/actor_critic_00001_00001_00001_00001/mountain_car/models --h11_w1 0.00001 --h11_w2 0.00001 --h12_w1 0.00001 --h12_w2 0.00001 --early_exit 90


python src/section3/main.py  \
--env CartPole-v1 \
--exp_name cartpole_exp \
--gamma 0.99 \
--oracle_1_env MountainCarContinuous-v0 \
--oracle_1_weights_path assets/section_1/actor_critic/mountain_car/models/mountain_car_exp/episode_best \
--oracle_2_env Acrobot-v1 \
--oracle_2_weights_path assets/section_1/actor_critic/cart_pole/models/cartpole_exp/episode_best \
--learning_rate_actor 0.00001 \
--learning_rate_critic 0.0001 \
--episodes 2000  \
--max_steps 2000 \
--models_dir assets/section_3/actor_critic_001_001_001_001/cartpole/models \
--h11_w1 0.001 \
--h11_w2 0.001 \
--h12_w1 0.001 \
--h12_w2 0.001 \
--early_exit 200000






python src/section3/main.py  \
--env CartPole-v1 \
--exp_name cartpole_exp \
--gamma 0.99 \
--oracle_1_env MountainCarContinuous-v0 \
--oracle_1_weights_path assets/section_1/actor_critic/mountain_car/models/mountaincar_exp/episode_best \
--oracle_2_env Acrobot-v1 \
--oracle_2_weights_path assets/section_1/actor_critic/acrobot/models/acrobot_exp/episode_best \
--learning_rate_actor 0.00001 \
--learning_rate_critic 0.0001 \
--episodes 2000  \
--max_steps 2000 \
--models_dir assets/section_3/actor_critic_001_001_001_001/cartpole/models \
--h11_w1 0.001 \
--h11_w2 0.001 \
--h12_w1 0.001 \
--h12_w2 0.001 \
-- optimize_hyper_parameters true \
--early_exit 2000






We used paramters similiar to ones we used when we trained the progresssive neural network on the mountain car.\\
We used same gamma,number of episodes,max steps, learning rate for actor and for critic.\\
We also implemen td the actor-critic method here the same as we did in section 1.\\
For the actor network we used a discrete policy estimaor that learned the size of the action space.\\
The policy network is built using proresswive nueral network where using a trained moutnrain car and acrobot model.
Their weights were frozen.
The architecture used for training the cartpole neural network is the same as the one used for training it in section 1.
We used used 4U adapters dense layers.
We saw that the activation that is coming out of the adapter had different scale those me multiplied activactions for 0.001.

For the critic network we had used both the critic models of "Acrobot-v1" and "MountainCarContinuous-v0" and we used a similar \\
architecture for the critic and similarity to the actor.\\
 we chose to multiply the activations by a constant. (h11\_w1,h12\_w1,h11\_w2,h12\_w2) of value 0.001 \\
as the activation values of the adapters were very big.