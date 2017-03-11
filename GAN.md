#### Adversarial nets

To learn the generator 's distribution $p_g$ over data $x$ ，we define a prior on input noise variables $p_z(z)$ .The $G(z; \theta_g) $ 

D and G paly the following two-player minimax game with value function $V(G,D)$ ：

$$\min\limits_{G}  \max\limits_{D}  V(D,G) = E_{x ~ p_{data} (x)} [log D(x)] + E _{z~ p_z (z)} [log(1-D(G(z)))]$$ 

* training critertion allows one to recover the data generating distribution as $G$ 
* D are given enough capacity in non-parametric limit

