# BMBC uncertainty

Implementation of S. Russo and P. Luchini, *A fast algorithm for the estimation
of statistical error in DNS (or experimental) time averages*, JCP 347 (2017)
328-340, https://doi.org/10.1016/j.jcp.2017.07.005. Equation numbers refer to
that paper.

The solver stores the means `X_k` of non-overlapping blocks of the 16 local
fields on the two centerlines, including the partial block, in
`statistics/bmbc_blocks.bin`.

```text
mean = (1/N) sum_n X_n,  N = M K                            (1)
X_k = mean of batch k,  x_k = X_k - mean                    (29)
variance(mean) = (S0 + 2 S1) / ((K - 1) (K - 2))            (30)
S0 = sum_{k=1..K} x_k^2                                     (31)
S1 = sum_{k=1..K-1} x_k x_{k+1}                             (32)
```

Equation (32) also prints `S1` as `sum (X_k X_{k+1} - mean^2)`. The two forms
differ by `-mean (x_1 + x_K)`; only the one above equals the correlation matrix
of centered terms in (28), which defines the estimator. The test suite checks
the implementation against that matrix.

The batch size `M` is the smallest one with `S1/S0 <= 0.5` (Section 5.2 and
Fig. 3), selected separately for every scalar series. It is scanned over the
multiples of the stored block that divide the number of blocks, so `N = M K`
holds exactly. Incomplete histories are first truncated to a multiple of 60
blocks, and the reported mean uses the same truncated samples. If no `M` with
at least three batches satisfies the limit, the uncertainty is `nan`, as it is
for a negative variance. An identically constant series has zero variance.

The normal distribution of the error gives the 95% interval
`mean +- 1.96 sqrt(variance)`.

Every uncertainty is that of a plain ensemble average. For the Reynolds
stresses and the turbulent dissipation the averaged variables are

```text
X = u'_i u'_j = (u_i - <u_i>)(u_j - <u_j>)
X = 2 nu s_ij s_ij,  s_ij = S_ij - <S_ij>
```

whose block means follow exactly from the stored block means, for example
`X_k = <u_i u_j>_k - <u_i><u_j>_k - <u_j><u_i>_k + <u_i><u_j>`. The intervals
of the RMS velocities and of the Kolmogorov scale are the images of the
intervals of `r_ii` and of the dissipation, both monotonic functions.

BMBC measures sampling uncertainty only; comparisons with Prasad and Koseff
also contain discretization, model, boundary-condition and experimental
differences.
