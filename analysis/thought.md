Suppose we are talking about having both friend rec and item rec, can we represent the user profile through the friend rec and item rec components. 
Meaning that the user preference is just a weighted sum of item rec and friend rec. and what we have is user/item id. 

So, we have the item side ID and the friend side ID. Meaning that how do we merge the ids for a final representation of the user. 

In another word, what token to take from item side and what token to take from the friend side. 

So for each of the token from the friend rec side, it is a combination of tokens from their friends. In other word, we can see the contribution of the friend's id in the final construction of the main user id? How do we denoise the social graph using the LLM. Then we can frame it as denoise, then rec.

what if we alternative the training process (idgen to model gen to idgen to model social )

### Model uncertainty calibration
so something pretty interesting is "use the friend information to calibrate the model's uncertainty". Since we are doing generative recommendation with constrained decoding, we can look at each prefix tree layer at a time. If a user have high probability on the model but it's 1-hop friends do not touch it, we can also make it decay as we go further into the social graph, then we will degrade the probability. Sure, maybe it would not change the final item prediction, but in some cases, it would. 

### Denoise by counterfactual training to simulate diffusion
Social influence kinda 'pollutes' the item preference space.
so let's try to simulate the effect of diffusion. Diffusion is basically adding noise until the input becomes pure noise, then the 'reverse' process is being guided by the recommendation loss. In our case, we have a generator and a recommender. If we think of the generator as the 'forward' process, then we can think of the recommender as the 'reverse' process

Forward process: Inject noise into the item/social ID. 
Suppose item-view ID is: [t1][t2][t3][t6] and social-view ID is [t1][t2][t4][t5]
We will save the common token [t1][t2]
We will construct noisy item-view ID by sampling n tokens from social-view id (number of tokens will be sampled, which token will be sampled). For example, noisy id [t3][t6]

We can sample n~Uniform(0, (length(id)-length(common token)))