# Theory Note

## 1. Multi-population LQ game and the observable fluctuation drift

Consider \(N\) interacting populations. For population \(i\), a representative state satisfies

$$
dX_t^{i,k}
=
\left(
a_i X_t^{i,k}
+
b_i \alpha_t^{i,k}
+
d_i
\right)dt
+
\sigma_i\, dW_t^{i,k}.
$$

The strategic network enters through the running cost rather than through the state equation. A convenient form is

$$
\begin{aligned}
f_i(x,\alpha,m)
&=
\frac{r_i}{2}\alpha^2
+
s_i x\alpha
+
\rho_i \alpha
+
\frac{q_i}{2}x^2
+
\beta_i x
-
x\, C_i^\top m
+
f_i^{(m)}(m),
\end{aligned}
$$

where \(m=(m_1,\dots,m_N)^\top\) is the vector of population means and \(C_i^\top\) denotes row \(i\) of the interaction matrix \(C\). Thus \(C_{ij}\neq 0\) means that population \(j\) enters the running cost of population \(i\). Cost-coupled linear-quadratic large-population games and their mean-field limits are standard in the LQG mean field game literature; see Huang, Caines, and Malhamé [1] and Bensoussan, Frehse, and Yam [2].

For an LQ mean field game, the value function is quadratic in the state. Writing the linear term in the value function as

$$
\ell(t)=S(t)m(t)+h(t),
$$

the optimal feedback is affine. Define

$$
M
=
\operatorname{diag}\!\left(\frac{b_i^2}{r_i}\right)
$$

and

$$
D_i(t)
=
\frac{b_i\bigl(b_i P_i(t)+s_i\bigr)}{r_i}
-
a_i,
$$

with \(D(t)\) diagonal. The matrix \(S(t)\) satisfies

$$
\dot S
=
D S + S D + S M S + C.
$$

The centered population fluctuations evolve linearly. If

$$
Z_t
=
\widehat m_t-\mathbb E[\widehat m_t],
$$

then, to leading order in the finite-population fluctuation scale,

$$
dZ_t
=
B(t) Z_t\,dt
+
\Gamma(t)\,dW_t,
$$

where

$$
B(t)
=
-D(t)-M S(t).
$$

This is the key observable quantity. The interaction matrix \(C\) is not read directly from the physical state equation; it is encoded in the drift of the equilibrium population fluctuations.

---

## 2. Identification of the strategic network

Starting from

$$
B=-D-MS,
$$

differentiate with respect to time:

$$
\dot B
=
-\dot D - M\dot S.
$$

Substituting the Riccati equation for \(S\),

$$
\dot S
=
DS+SD+SMS+C,
$$

gives

$$
\dot B
=
-\dot D
-
MDS
-
MSD
-
MSMS
-
MC.
$$

Because \(M\) and \(D\) are diagonal, they commute. Using

$$
MS=-(B+D),
$$

we obtain

$$
MDS
=
DMS
=
-D(B+D),
$$

$$
MSD
=
-(B+D)D,
$$

and

$$
MSMS
=
(B+D)^2.
$$

After collecting terms,

$$
MC
=
-\dot B
-\dot D
-B^2
+D^2.
$$

The diagonal matrices \(D\), \(\dot D\), and \(D^2\) contribute only to diagonal entries. Therefore, for \(i\neq j\),

$$
(MC)_{ij}
=
-\dot B_{ij}
-
(B^2)_{ij}.
$$

It is useful to define the identifiable scaled network

$$
G
=
MC.
$$

Then the off-diagonal recovery formula is

$$
\boxed{
G_{ij}
=
-\dot B_{ij}
-
(B^2)_{ij},
\qquad i\neq j.
}
$$

This identity is the project-specific inverse relation used throughout the code and experiments; it follows directly from the LQ equilibrium equations above.

In the default numerical configuration,

$$
r_i=b_i^2,
$$

so \(M=I\) and therefore

$$
G=C.
$$

More generally, if \(M\) is known, \(C\) is recovered by row rescaling:

$$
C=M^{-1}G.
$$

If \(M\) is unknown but positive diagonal, the exact row scale is not identified from the fluctuation drift alone. The support pattern and the sign of each off-diagonal entry are still preserved because multiplying a row by a positive scalar does not change either.

---

## 3. Estimation from repeated finite-population trajectories

Suppose \(R\) independent realizations of the finite-player system are observed on the grid

$$
t_n=n\Delta t,
\qquad n=0,\dots,N_T.
$$

Let \(\widehat m_r(t_n)\) denote the empirical population mean in run \(r\). The centered panel is

$$
Z_r(t_n)
=
\widehat m_r(t_n)
-
\frac{1}{R}
\sum_{s=1}^{R}
\widehat m_s(t_n).
$$

The local fluctuation model is

$$
\Delta Z_{r,n}
=
B(t_n)Z_r(t_n)\Delta t
+
\text{innovation}.
$$

### Pointwise OLS

The pointwise estimator fits \(B(t_n)\) separately at every observation time by regressing

$$
\frac{\Delta Z_{r,n}}{\Delta t}
$$

on

$$
Z_r(t_n)
$$

across the \(R\) independent runs.

After estimating \(B(t_n)\), the time derivative is approximated by centered finite differences and the network estimate is formed from

$$
\widehat G_{ij}
=
-\widehat{\dot B}_{ij}
-
(\widehat B^2)_{ij}.
$$

This method is direct, but each local regression uses only the cross-sectional sample available at one time. When \(R\) is small, the noise in \(\widehat B(t_n)\) can be substantial.

The quadratic term is especially important. If

$$
\widehat B=B+E,
$$

then

$$
\widehat B^2
=
B^2
+
BE
+
EB
+
E^2.
$$

The final \(E^2\) contribution is a same-sample second-order term. Even when the first-order error is centered, \(E^2\) does not average away in the same manner.

### Cross-fitted pointwise OLS

To reduce this quadratic self-interaction, the runs are split into two disjoint folds. Let the corresponding drift estimates be \(\widehat B^{A}\) and \(\widehat B^{B}\). The quadratic term is replaced by

$$
\widehat{B^2}_{\mathrm{CF}}
=
\frac{1}{2}
\left(
\widehat B^{A}\widehat B^{B}
+
\widehat B^{B}\widehat B^{A}
\right).
$$

Because the two factors are fit from independent groups of trajectories, the leading same-sample quadratic term is removed. This matters most when the local drift estimates are noisy. Cross-fitting and sample splitting are standard devices for separating estimation error from the observations used in a subsequent estimating equation; see Chernozhukov et al. [4]. The particular symmetrized matrix product above is specific to the quadratic \(B^2\) term in this problem.

### Martingale regression

The alternative used in the project is a globally pooled parametric drift estimate. Write

$$
B(t)
\approx
\sum_{k=0}^{3}
\Theta_k \phi_k(t),
$$

where \(\phi_k\) are cubic Legendre basis functions after mapping time to \([-1,1]\).

The coefficients are estimated from all runs and all time increments at once through the diffusion regression

$$
\Delta Z_{r,n}
\approx
\left(
\sum_{k=0}^{3}
\Theta_k \phi_k(t_n)
\right)
Z_r(t_n)\Delta t
+
\text{innovation}.
$$

The regressors are predictable with respect to the filtration generated by the observed process, while the Brownian innovation has conditional mean zero. This is the standard setting for martingale estimating functions for diffusion processes; see Sørensen [3]. The cubic Legendre representation itself is a modeling choice made in this project rather than a requirement of that theory.

The practical advantage is that the drift is not re-estimated independently at every time point. Information is pooled across the full horizon, and

$$
\dot B(t)
=
\sum_{k=0}^{3}
\Theta_k \dot\phi_k(t)
$$

is available analytically. The network estimate then uses the same structural identity,

$$
\widehat G_{ij}
=
-\widehat{\dot B}_{ij}
-
(\widehat B^2)_{ij}.
$$

A cross-fitted martingale version is obtained by fitting the basis coefficients on two trajectory folds and using the same symmetrized cross-product for \(B^2\). In practice the gain is usually smaller than for pointwise OLS because the pooled martingale estimator already has much lower drift-estimation variance.

---

## 4. Infinite horizon

In the stationary problem, the equilibrium fluctuation drift is constant:

$$
B(t)=B_\infty.
$$

Hence

$$
\dot B_\infty=0,
$$

and the off-diagonal identification formula reduces to

$$
\boxed{
G_{ij}
=
-(B_\infty^2)_{ij},
\qquad i\neq j.
}
$$

The centered process is an Ornstein-Uhlenbeck diffusion,

$$
dZ_t
=
B_\infty Z_t\,dt
+
\Gamma\,dW_t,
$$

with \(B_\infty\) required to be Hurwitz for a stable stationary equilibrium. Statistical inference for ergodic diffusions, including OU-type models, is treated systematically by Kutoyants [5].

### Exact OU transition regression

For a fixed observation interval \(\Delta t\),

$$
Z_{n+1}
=
A Z_n+\varepsilon_n,
$$

where

$$
A=e^{B_\infty \Delta t}.
$$

The exact transition matrix is estimated by multivariate least squares,

$$
\widehat A
=
\left(
\sum Z_{n+1}Z_n^\top
\right)
\left(
\sum Z_n Z_n^\top
\right)^{-1},
$$

followed by

$$
\widehat B_\infty
=
\frac{1}{\Delta t}\log(\widehat A).
$$

The network estimate is then

$$
\widehat G_{ij}
=
-(\widehat B_\infty^2)_{ij}.
$$

Using the exact Gaussian transition of a multidimensional Ornstein-Uhlenbeck process is classical; see Pap and van Zuijlen [6]. The advantage here is that the conditional mean is represented exactly rather than through an Euler step.

### Continuous-time regression

The second stationary estimator uses

$$
\Delta Z_n
\approx
B_\infty Z_n\,\Delta t
+
\text{innovation},
$$

which gives

$$
\widehat B_\infty
=
\left(
\sum \Delta Z_n Z_n^\top
\right)
\left(
\Delta t
\sum Z_n Z_n^\top
\right)^{-1}.
$$

This is the discrete implementation of the natural continuous-time drift estimating equation for a linear ergodic diffusion; see Kutoyants [5] for the broader likelihood and asymptotic theory.

In the infinite-horizon experiments, \(T\) should be interpreted as observation duration, not as a game terminal time.

---

## 5. Error scaling and interpretation

The numerical experiments are organized around the number of independent runs \(R\) and the observation horizon \(T\).

Empirically, once both are sufficiently large, the recovery error is consistent with the first-order scaling

$$
\operatorname{error}
\asymp
\frac{1}{\sqrt{RT}}.
$$

This is the natural rate suggested by pooling information across independent trajectories and over time. The finite-sample constants, however, differ substantially across estimators.

For small \(R\), pointwise OLS can remain noisy even when \(T\) is large because every local drift estimate is built from a limited cross section. Cross-fitting reduces the quadratic self-noise in this regime. The martingale estimator performs better because it uses the entire trajectory panel in a single low-dimensional drift fit.

At large \(R\) and \(T\), the distinction between the methods becomes less dramatic. Their first-order behavior is similar, while the main differences are finite-sample variance and second-order bias.

The relation

$$
\operatorname{error}
\asymp
(RT)^{-1/2}
$$

should be understood here as an empirical scaling law. A formal theorem would require explicit assumptions on stability, excitation, dependence over time, and the asymptotic regime. Time increments within a single trajectory are not independent, and the centering step also introduces a small dependence across runs at finite \(R\). Those effects do not change the structural recovery identity, but they matter for a fully rigorous sampling theory.

The main identification point is independent of those statistical details: once the equilibrium fluctuation drift \(B\) is learned accurately enough, the off-diagonal strategic network is recovered from a deterministic algebraic identity.

---

## References

[1] M. Huang, P. E. Caines, and R. P. Malhamé, “Large-Population Cost-Coupled LQG Problems With Nonuniform Agents: Individual-Mass Behavior and Decentralized \(\varepsilon\)-Nash Equilibria,” *IEEE Transactions on Automatic Control*, 52(9), 1560–1571, 2007. https://doi.org/10.1109/TAC.2007.904450

[2] A. Bensoussan, J. Frehse, and P. Yam, *Mean Field Games and Mean Field Type Control Theory*. Springer, 2013. https://doi.org/10.1007/978-1-4614-8508-7

[3] M. Sørensen, “Estimating Functions for Diffusion-Type Processes,” in *Statistical Methods for Stochastic Differential Equations*, M. Kessler, A. Lindner, and M. Sørensen, eds., CRC Press, 2012, pp. 1–107.

[4] V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins, “Double/Debiased Machine Learning for Treatment and Structural Parameters,” *The Econometrics Journal*, 21(1), C1–C68, 2018. https://doi.org/10.1111/ectj.12097

[5] Y. A. Kutoyants, *Statistical Inference for Ergodic Diffusion Processes*. Springer, 2004. https://doi.org/10.1007/978-1-4471-3866-2

[6] G. Pap and M. C. A. van Zuijlen, “Parameter Estimation with Exact Distribution for Multidimensional Ornstein-Uhlenbeck Processes,” *Journal of Multivariate Analysis*, 59(2), 153–165, 1996. https://doi.org/10.1006/jmva.1996.0058
