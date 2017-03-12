#### Wasserstein GAN

The paper is concerned with unsupervised learning.  The classical way is solve the problem:
$ \max\limits_{\theta \in R^d} \frac{1}{m} \sum\limits_{i=1} ^m log P_{\theta}(x^{(i)}) $   

the real data distribution $\mathbb{P}_r$  and the distribution of the parametrized density $\mathbb{P}_{\theta}$ 

问题等价于$KL(\mathbb{P}_r || \mathbb{P_{\theta}})$ 

if $\rho$ is our notion of distance between two distributions, we would like to have a loss function $\theta \mapsto\rho( \mathbb{P}_{\theta},\mathbb{P}_r)$ 

##### contributions

*  Earth Mover (EM) distance behaves in comparison to popular probability distances
* Define a form of GAN called Wasserstein-GAN that minimizes a reasonable and efficient approximation of the EM distance			  



The total variation (TV) distance:

$\delta(\mathbb{P}_r,\mathbb{P}_g) = \sum\limits_{A\in \Sigma} |\mathbb{P}_r(A) - \mathbb{P}_g(A) |$ 

The kullback-Leibler divergence:

$KL(\mathbb{P}_r || \mathbb{P}_g) = \int log(\frac{P_r(x)}{P_g(x)}) P_r(x) d\mu(x)$

The Jensen-Shannon divergence:

$JS(\mathbb{P}_r,\mathbb{P}_g) = KL(\mathbb{P}_r || \mathbb{P}_m) + KL(\mathbb{P}_g || \mathbb{P}_m)$

The Earth-Mover(EM) distance or Wasserstein-1:

$W(\mathbb{P}_r,\mathbb{P}_g) = \inf\limits_{\gamma \in \prod (\mathbb{P}_r,\mathbb{P}_g)} E_{(x,y) \sim \gamma}[||x-y||]$



* Theorem 3. Let $\mathbb{P}_r$ be any distribution. Let $\mathbb{P}_{\theta}$ be the distribution of $g_{\theta}(Z)$ with $Z$ a random variable with density p and $g_{\theta}$ a function satisfying assumption. Then there is a solution $f : \cal X \rightarrow \mathbb{R} $ to the problem

  $$\max\limits_{||f||_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}_r} [f(x)] - \mathbb{E}_{x \sim \mathbb{P}_{\theta} }[f(x)]$$

  and we have 

  $$\nabla_{\theta} W(\mathbb{P}_r, \mathbb{P}_{\theta}) = -\mathbb{E}_{z \sim p(z)}[\nabla_{\theta}f(g_{\theta}(z))] $$

  when both terms are well defined

