#### Adversarial nets

To learn the generator 's distribution $p_g$ over data $x$ ，we define a prior on input noise variables $p_z(z)$ .The $G(z; \theta_g) $ 

D and G paly the following two-player minimax game with value function $V(G,D)$ ：

$$\min\limits_{G}  \max\limits_{D}  V(D,G) = E_{x ~ p_{data} (x)} [log D(x)] + E _{z~ p_z (z)} [log(1-D(G(z)))]$$ 

* training critertion allows one to recover the data generating distribution as $G$ 
* D are given enough capacity in non-parametric limit


在训练时候出现的over-fitting问题：

应该要between k steps of optimizing D and one step of optimizing G

$x = G(z)$ ，是在domain of z 上进行采样，并且得到non-uniform distribution $p_g$ on the transformed samples.

最后这个对抗学习过程需要达到一个收敛：

$p_g$  is similar to $p_{data}$  and D is a partiallt accurate classifier.

最后两个都达到各自理想的效果。

在这个过程中主要是通过$G(z)$ 的梯度下降算法来进行

$D(G) \rightarrow \frac{p_{data}(x)} {p_{data}(x) + p_g(x)} $  到最后会达到$\frac{1}{2}$ 



* Update the discriminator by ascending its stochastic gradient:

  $$\nabla_{\theta_d} \frac{1}{m} \sum\limits_{i=1}^m [ logD(x^{(i)}) + log(1-D(G(z^{(i)})))]$$

* Update the generator by descending its stochastic gradient:

  $$\nabla_{\theta_g} \frac{1}{m} \sum\limits_{i=1}^m log(1-D(G(z^{(i)}))) $$



##### Global Optimality of $p_g = p_{data}$ 

proposition1. For G fixed, the optimal discriminator D is 

$G_G^*(x) = \frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$ 

* Proof: the training criterion for the discriminator D, given any generator G, is to maximize the quantity $V(G,D)$ 

  $\begin{align}  V(G,D)=&\int_x p_{data}(x)log(D(x)) + \int_x p_z(z)log(1-D(g(z)))dz\\ =&\int_xp_{data}(x)log(D(x))+p_g(x)log(1-D(x))dx \end{align}$



$$C(G) = E_{x~p_{data}} [log \frac{p_{data}(x)}{p_{data}(x)+p_g(x)}] + E_{x~p_g}[log\frac{p_g(x)}{p_{data}(x)+p_g(x)}] $$

 

