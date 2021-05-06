---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: elections-models
    language: python
    name: elections-models
---

# Popularity hide and seek
> "A hidden Markov model to estimate latent presidential popularity across time"

- toc: true
- badges: true
- comments: true
- author: Alexandre Andorra, Rémi Louf
- categories: [popularity, Macron, Hidden Markov models, polls]
- image: images/gp-popularity.png


A few months ago, [I experimented with a Gaussian Process](https://alexandorra.github.io/pollsposition_blog/popularity/macron/gaussian%20processes/polls/2021/01/18/gp-popularity.html) to estimate the popularity of French presidents across time. The experiment was really positive, and helped me get familiar with the beauty of GPs. This time, I teamed up with [Rémi Louf](https://twitter.com/remilouf) on a [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) to estimate the same process -- what is the true latent popularity, that we only observe through the noisy data that are polls?

This was supposed to be a trial run before working on an electoral model for the coming regional elections in France -- it's always easier to start with 2 dimensions than 6, right? But the model turned out to be so good at smoothing and predicting popularity data that we thought it'd be a shame not to share it. And voilà!

## Show me the data!

The data are the same as in [my GP post](https://alexandorra.github.io/pollsposition_blog/popularity/macron/gaussian%20processes/polls/2021/01/18/gp-popularity.html), so we're not going to spend a lot of time explaining them. It's basically all the popularity opinion polls of French presidents since the term limits switched to 5 years (in 2002).

Let's import those data, as well as the (fabulous) packages we'll need:

```python
import datetime

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as aet
from scipy.special import expit as logistic
```

```python
# hide
RANDOM_SEED = 926
np.random.seed(RANDOM_SEED)
arviz.style.use("arviz-darkgrid")
```

```python
data = pd.read_csv(
    "https://raw.githubusercontent.com/AlexAndorra/pollsposition_models/master/data/raw_popularity_presidents.csv",
    header=0,
    index_col=0,
    parse_dates=True,
)
```

```python
# hide
# restrict data to after the switch to 5-year term
data = data.loc[data.index >= pd.to_datetime("2002-05-05")]
data["year"] = data.index.year
data["month"] = data.index.month

data["sondage"] = data["sondage"].replace("Yougov", "YouGov")
data["method"] = data["method"].replace("face-to-face&internet", "face to face")

# convert to proportions
data[["approve_pr", "disapprove_pr"]] = data[["approve_pr", "disapprove_pr"]].copy() / 100
data = data.rename(columns={"approve_pr": "p_approve", "disapprove_pr": "p_disapprove"})

POLLSTERS = data["sondage"].sort_values().unique()
comment = f"""The dataset contains {len(data)} polls between the years {data["year"].min()} and {data["year"].max()}.
There are {len(POLLSTERS)} pollsters: {', '.join(list(POLLSTERS))}
"""
print(comment)
```

The number of polls is homogeneous among months, except in the summer because, well, France:

```python
data["month"].value_counts().sort_index()
```

Let us look at simple stats on the pollsters:

```python
pd.crosstab(data.sondage, data.method, margins=True)
```

Interesting: most pollsters only use one method -- internet. Only BVA, Ifop, Ipsos (and Kantar very recently) use different methods. So, if we naively estimate the biases of pollsters and methods individually, we'll get high correlations in our posterior estimates -- the parameter for `face to face` will basically be the one for `Kantar`, and vice versa. So we will need to model the pairs `(pollster, method)` rather than pollsters and methods individually.

Now, let's just plot the raw data and see what they look like:

```python
approval_rates = data["p_approve"].values
disapproval_rates = data["p_disapprove"].values
doesnotrespond = 1 - approval_rates - disapproval_rates
newterm_dates = data.reset_index().groupby("president").first()["index"].values
dates = data.index

fig, axes = plt.subplots(3, figsize=(12, 8))
for ax, rate, label in zip(axes.ravel(), [approval_rates, disapproval_rates, doesnotrespond], ["Approve", "Disapprove", "No answer"]):
    ax.plot(dates, rate, "o", alpha=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel(label)
    for date in newterm_dates:
        ax.axvline(date, color="k", alpha=0.6, linestyle="--")
```

We notice two things when looking at these plots:

1. Approval rates systematically decrease as the goes on.
2. While that's true, some events seem to push the approval rate back up, even though temporarily. This happened in every term, actually. Can that variance really be explained solely with a random walk?
3. Non-response rate is higher during Macron's term.


## Monthly standard deviation

Something that often proves challenging with count data is that they are often more dispersed than traditional models expect them to be. Let's check this now, by computing the monthly standard deviation of the approval rates (we weigh each poll equally, even though we probably should weigh them according to their respective sample size):

```python
rolling_std = (
    data.reset_index()
    .groupby(["year", "month"])
    .std()
    .reset_index()[["year", "month", "p_approve"]]
)
rolling_std
```

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pd.to_datetime([f'{y}-{m}-01' for y, m in zip(rolling_std.year, rolling_std.month)]), rolling_std.p_approve.values, "o", alpha=0.5)
ax.set_title("Monthly standard deviation in polls")
for date in newterm_dates:
    ax.axvline(date, color="k", alpha=0.6, linestyle="--")
```

There is a very high variance for Chirac's second term, and for the beggining of Macron's term. For Chirac's term, it seems like the difference stems from the polling method: face-to-face approval rates seem to be much lower. For Macron, this high variance is quite hard to explain. In any case, we'll probably have to take this overdispersion (as it's called in statistical linguo) of the data in our models...

```python
face = data[data["method"] == "face to face"]
dates_face = face.index

other = data[data["method"] != "face to face"]
dates_other = other.index

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates_face, face["p_approve"].values, "o", alpha=0.3, label="face to face")
ax.plot(dates_other, other["p_approve"].values, "o", alpha=0.3, label="other")
ax.set_ylim(0, 1)
ax.set_ylabel("Does approve")
ax.set_title("Raw approval polls")
ax.legend()
for date in newterm_dates:
    ax.axvline(date, color="k", alpha=0.6, linestyle="--")
```

## A raw analysis of bias

As each pollster uses different methods to establish and question their samples each month, we don't expect their results to be identical -- _that_ would be troubling. Instead we expect each pollster and each polling method to be at a different place on the spectrum: some report popularity rates in line with the market average, some are below average, some are above.

The model will be able to estimate this bias on the fly and more seriously (if we tell it to), but let's take a look at a crude estimation ourselves, to get a first idea. Note that we're talking about _statistical_ bias here, not _political_ bias: it's very probable that reaching out to people only by internet or phone can have a [selection effect](https://en.wikipedia.org/wiki/Selection_bias) on your sample, without it being politically motivated -- statistics are just hard and stubborn you know 🤷‍♂️

To investigate bias, we now compute the monthly mean of the $p_{approve}$ values and check how each individual poll strayed from this mean:

```python
data = (
    data.reset_index()
    .merge(
        data.groupby(["year", "month"])["p_approve"].mean().reset_index(),
        on=["year", "month"],
        suffixes=["", "_mean"],
    )
    .rename(columns={"index": "field_date"})
)
data["diff_approval"] = data["p_approve"] - data["p_approve_mean"]
data.round(2)
```

Then, we can aggregate these offsets by pollster and look at their distributions:

```python
POLLSTER_VALS = {
    pollster: data[data["sondage"] == pollster]["diff_approval"].values
    for pollster in list(POLLSTERS)
}

colors = plt.rcParams["axes.prop_cycle"]()
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(12, 12))

for ax, (pollster, vals) in zip(axes.ravel(), POLLSTER_VALS.items()):
    c = next(colors)["color"]
    ax.hist(vals, alpha=0.3, color=c, label=pollster)
    ax.axvline(x=np.mean(vals), color=c, linestyle="--")
    ax.axvline(x=0, color="black")
    ax.set_xlim(-0.3, 0.3)
    ax.legend()

plt.xlabel(r"$p_{approve} - \bar{p}_{approve}$", fontsize=25);
```

A positive (resp. negative) bias means the pollster tends to report higher (resp. lower) popularity rates than the average pollster. We'll see what the model has to say about this, but our prior is that, for instance, YouGov and Kantar tend to be below average, while Harris and BVA tend to be higher.

And now for the bias per method:

```python
METHOD_VALS = {
    method: data[data["method"] == method]["diff_approval"].values
    for method in list(data["method"].unique())
}

colors = plt.rcParams["axes.prop_cycle"]()
fig, ax = plt.subplots(figsize=(11, 5))

for method, vals in METHOD_VALS.items():
    c = next(colors)["color"]
    ax.hist(vals, alpha=0.3, color=c, label=method)
    ax.axvline(x=np.mean(vals), color=c, linestyle="--")

ax.axvline(x=0, color="black")
ax.set_xlim(-0.2, 0.2)
ax.set_xlabel(r"$p_+ - \bar{p}_{+}$", fontsize=25)
ax.legend();
```

Face-to-face polls seem to give systematically below-average approval rates, while telephone polls seem to give slightly higher-than-average results.

Again, keep in mind that there is substantial correlation between pollsters and method, so take this with a grain of salt -- that's why it's useful to add that to the model actually: it will be able to decipher these correlations, integrate them into the full data generating process, and report finer estimates of each bias. 

Speaking of models, do you know what time it is? It's model time, of course!!


## Model

Each poll $i$ at month $m$ from the beginning of a president’s term finds that
$y_i$ individuals have a positive opinion of the president’s action over
$n_i$ respondents. We model this as

$$y_{i,m} \sim Binomial(p_{i,m}, n_{i,m})$$

We loosely call $p_{i,m}$ the *popularity* of the president, $m$ month into his
presidency. This is the quantity we would like to model.

Why specify the month when the time information is already contained in the
succession of polls? Because French people tend to be less and less satisfied
with their president as their term moves, regardless of their action.

We model $p_{i,m}$ with a random walk logistic regression:

$$p_{i,m} = logit^{-1}(\mu_m + \alpha_k + \zeta_j)$$

$\mu_m$ is the underlying support for the president at month $m$. $\alpha_k$ is
the bias of the pollster, while $\zeta_j$ is the inherent bias of the polling
method. The biases are assumed to be completely unpooled at first, i.e we model
one bias for each pollster and method:

$$\alpha_k \sim Normal(0, \sigma_k)\qquad \forall pollster k$$

and 

$$\zeta_j \sim Normal(0, \sigma_j)\qquad \forall method j$$

We treat the time variation of $\mu$ with a correlated random walk:

$$\mu_m | \mu_{m-1} \sim Normal(\mu_{m-1}, \sigma_m)$$

For the sake of simplicity, we choose not to account at first for a natural
decline in popularity $\delta$, the unmeployment at month $m$, $U_m$, or
random events that can happen during the term. 

```python
data["num_approve"] = np.floor(data["samplesize"] * data["p_approve"]).astype("int")
data
```

Each observation is uniquely identified by `(pollster, field_date)`:

```python
pollster_by_method_id, pollster_by_methods = data.set_index(
    ["sondage", "method"]
).index.factorize(sort=True)
month_id = np.hstack(
    [
        pd.Categorical(
            data[data.president == president].field_date.dt.to_period("M")
        ).codes
        for president in data.president.unique()
    ]
)
months = np.arange(max(month_id) + 1)
```

```python
COORDS = {
    "pollster_by_method": pollster_by_methods,
    "month": months,
    "observation": data.set_index(["sondage", "field_date"]).index,
}
```

### Fixed `mu` for GRW

```python
with pm.Model(coords=COORDS) as pooled_popularity:

    bias = pm.Normal("bias", 0, 0.15, dims="pollster_by_method")
    mu = pm.GaussianRandomWalk("mu", sigma=1.0, dims="month")

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(mu[month_id] + bias[pollster_by_method_id]),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )

    idata = pm.sample(return_inferencedata=True)
```

We plot the posterior distribution of the pollster and method biases:

```python
arviz.plot_trace(idata, var_names=["~popularity"], compact=True);
```

Since we are performing a logistic regression, these coefficients can be tricky to interpret. When the bias is positive, this means that we need to add to the latent popularity to get the observation, which means that the pollster/method tends to be biased towards giving higher popularity scores.

```python
arviz.summary(idata, round_to=2, var_names=["~popularity"])
```

```python
mean_bias = idata.posterior["bias"].mean(("chain", "draw")).to_dataframe()
mean_bias.round(2)
```

```python
ax = mean_bias.plot.bar(figsize=(14, 8), rot=30)
ax.set_title("$>0$ bias means (pollster, method) overestimates the latent popularity");
```

We now plot the posterior values of `mu`. Since the model is completely pooled, we only have 60 values, which correspond to a full term:

```python
post_pop = logistic(idata.posterior["mu"].stack(sample=("chain", "draw")))

fig, ax = plt.subplots()
for i in np.random.choice(post_pop.coords["sample"].size, size=1000):
    ax.plot(
        idata.posterior.coords["month"],
        post_pop.isel(sample=i),
        alpha=0.01,
        color="blue",
    )
post_pop.mean("sample").plot(ax=ax, color="orange", lw=2)
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

```python
hdi_data = arviz.hdi(logistic(idata.posterior["mu"]))
ax = arviz.plot_hdi(idata.posterior.coords["month"], hdi_data=hdi_data)
ax.vlines(
    idata.posterior.coords["month"],
    hdi_data.sel(hdi="lower")["mu"],
    hdi_data.sel(hdi="higher")["mu"],
)
post_pop.median("sample").plot(ax=ax);
```

```python
arviz.plot_posterior(logistic(idata.posterior["mu"].sel(month=42)));
```

### Infer the standard deviation $\sigma$ of the random walk

```python
with pm.Model(coords=COORDS) as pooled_popularity:

    bias = pm.Normal("bias", 0, 0.15, dims="pollster_by_method")
    sigma_mu = pm.HalfNormal("sigma_mu", 0.5)
    mu = pm.GaussianRandomWalk("mu", sigma=sigma_mu, dims="month")

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(mu[month_id] + bias[pollster_by_method_id]),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )

    idata = pm.sample(tune=2000, draws=2000, return_inferencedata=True)
```

```python
arviz.plot_trace(idata, var_names=["~popularity"], compact=True);
```

```python
arviz.summary(idata, round_to=2, var_names=["~popularity"])
```

```python
post_pop = logistic(idata.posterior["mu"].stack(sample=("chain", "draw")))

fig, ax = plt.subplots()
for i in np.random.choice(post_pop.coords["sample"].size, size=1000):
    ax.plot(
        idata.posterior.coords["month"],
        post_pop.isel(sample=i),
        alpha=0.01,
        color="blue",
    )
post_pop.mean("sample").plot(ax=ax, color="orange", lw=2)
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

```python
hdi_data = arviz.hdi(logistic(idata.posterior["mu"]))
ax = arviz.plot_hdi(idata.posterior.coords["month"], hdi_data=hdi_data)
ax.vlines(
    idata.posterior.coords["month"],
    hdi_data.sel(hdi="lower")["mu"],
    hdi_data.sel(hdi="higher")["mu"],
)
post_pop.median("sample").plot(ax=ax);
```

The posterior variance of the values of $\mu$ looks grossly underestimated; between month 40 and 50 presidents have had popularity rates between .2 nd .4 while here the popularity is estimated aournd .21 plus or minus .02 at best. We need to fhix this.


### A model that accounts for the overdispersion of polls


As we saw with the previous model, the variance of $\mu$'s posterior values is grossly underestimated. This suggests that the variance in the obervations is not only due to variations in the mean value, $p_{approve}$. Indeed, there is variance in the results that probably cannot be accounted for by the pollsters' and method's biases and has more something to do with measurement errors, or other factors we did not include.

We use a Beta-Binomial model to add one degree of liberty and allow the variance to be estimated independently from the mean value:

```python
with pm.Model(coords=COORDS) as pooled_popularity:

    bias = pm.Normal("bias", 0, 0.15, dims="pollster_by_method")
    sigma_mu = pm.HalfNormal("sigma_mu", 0.5)
    mu = pm.GaussianRandomWalk("mu", sigma=sigma_mu, dims="month")

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(mu[month_id] + bias[pollster_by_method_id]),
        dims="observation",
    )

    # overdispersion parameter
    theta = pm.Exponential("theta_offset", 1.0) + 10.0

    N_approve = pm.BetaBinomial(
        "N_approve",
        alpha=popularity * theta,
        beta=(1.0 - popularity) * theta,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )

    idata = pm.sample(tune=2000, draws=2000, return_inferencedata=True)
```

```python
arviz.plot_trace(idata, var_names=["~popularity"], compact=True);
```

```python
arviz.summary(idata, round_to=2, var_names=["~popularity"])
```

```python
post_pop = logistic(idata.posterior["mu"].stack(sample=("chain", "draw")))

fig, ax = plt.subplots()
for i in np.random.choice(post_pop.coords["sample"].size, size=1000):
    ax.plot(
        idata.posterior.coords["month"],
        post_pop.isel(sample=i),
        alpha=0.01,
        color="blue",
    )
post_pop.mean("sample").plot(ax=ax, color="orange", lw=2)
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

```python
hdi_data = arviz.hdi(logistic(idata.posterior["mu"]))
ax = arviz.plot_hdi(idata.posterior.coords["month"], hdi_data=hdi_data)
ax.vlines(
    idata.posterior.coords["month"],
    hdi_data.sel(hdi="lower")["mu"],
    hdi_data.sel(hdi="higher")["mu"],
)
post_pop.median("sample").plot(ax=ax);
```

This is much better! It is unlikely we would be able to do much better than this for the unpooled model; maybe by having one dispersion term per term/month. But since we wish to switch to a partially pooled model for $\mu$ we will stop our investigation on the fully pooled model for now.


### Hierarchical model

```python
president_id, presidents = data["president"].factorize(sort=False)
COORDS["president"] = presidents
```

```python
with pm.Model(coords=COORDS) as hierarchical_popularity:

    house_effect = pm.Normal("house_effect", 0, 0.15, dims="pollster_by_method")
    month_effect = pm.Normal("month_effect", 0, 0.15, shape=len(COORDS["month"]) + 1)
    shrinkage_pop = pm.HalfNormal("shrinkage_pop", 0.2)
    month_president_effect = pm.GaussianRandomWalk(
        "month_president_effect",
        mu=month_effect,
        sigma=shrinkage_pop,
        dims=("president", "month"),
    )

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
            month_president_effect[president_id, month_id]
            + house_effect[pollster_by_method_id]
        ),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )
```

```python
with hierarchical_popularity:
    idata = pm.sample(return_inferencedata=True)
```

Mmmh, that doesn't sample because we get zero derivates for some variables... Let's check the model's test point:

```python
hierarchical_popularity.check_test_point()
```

Interesting: the problem doesn't come from a -inf test point or from missing values in the data -- the problem really comes from the model. A safe bet here is to try and reparametrize the model with a non-centered parametrization:

```python
with pm.Model(coords=COORDS) as hierarchical_popularity:

    house_effect = pm.Normal("house_effect", 0, 0.15, dims="pollster_by_method")

    month_effect = pm.Normal("month_effect", 0, 0.15, dims="month")
    sd = pm.HalfNormal("shrinkage_pop", 0.2)
    raw_rw = pm.GaussianRandomWalk("raw_rw", sigma=1.0, dims=("president", "month"))
    month_president_effect = pm.Deterministic(
        "month_president_effect",
        month_effect + raw_rw * sd,
        dims=("president", "month"),
    )

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
            month_president_effect[president_id, month_id]
            + house_effect[pollster_by_method_id]
        ),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )

    idata = pm.sample(return_inferencedata=True)
```

So our sampling problem indeed came from a challenging geometry when the model was parametrized with the centered parametrization. Switching to a non-centered one fixed it. Now, do our estimates make sense?

```python
arviz.plot_trace(
    idata,
    var_names=["~popularity", "~rw"],
    filter_vars="regex",
);
```

That looks a bit weird right? `shrinkage_pop`, the random walk's standard deviation, seems really high! That's basically telling us that the president's popularity can change a lot from one month to another, which we now from domain knowledge is not true. The `month_effect` are all similar and centered on 0, which means all months are very similar -- there can't really be a bad month or a good month. 

This is worrying for at least two reasons: 1) we _know_ from prior knowledge that there _are_ good and bad months for presidents; 2) this extreme similarity in `month_effect` directly contradicts the high `shrinkage_pop`: how can the standard deviation be so high if months are all the same?

So something is missing here. Actually, we should really have an intercept, which represents the baseline presidential approval, no matter the month and president. The tricky thing here is that `pm.GaussianRandomWalk` uses [a distribution to initiate the random walk](https://docs.pymc.io/api/distributions/timeseries.html#pymc3.distributions.timeseries.GaussianRandomWalk). So, if we don't constrain it to zero, we will get an additive non-identifiability -- for each president and month, we'll have two intercepts, `baseline` and the initial value of the random walk. `pm.GaussianRandomWalk` only accepts distribution objects for the `init` kwarg though, so we have to implement the random walk by hand, i.e:

$$\mu_n = \mu_{n - 1} + Z_n, \, with \, Z_n \sim Normal(0, 1) \, and \, \mu_0 = 0$$

In other words, a Gaussian random walk is just a cumulative sum, where we add a sample from a standard Normal at each step ($Z_n$ here, which is called the innovation of the random walk).

Finally, it's probably useful to add a `president_effect`: it's very probable that some presidents are just more popular than others, even when taking into account the cyclical temporal variations.

Let's code that up!

```python
COORDS["month_minus_origin"] = COORDS["month"][1:]
```

```python
with pm.Model(coords=COORDS) as hierarchical_popularity:

    baseline = pm.Normal("baseline")
    president_effect = pm.Normal("president_effect", sigma=0.15, dims="president")
    house_effect = pm.Normal("house_effect", 0, 0.15, dims="pollster_by_method")

    month_effect = pm.Normal("month_effect", 0, 0.15, dims="month")
    # need the cumsum parametrization to properly control the init of the GRW
    rw_init = aet.zeros(shape=(len(COORDS["president"]), 1))
    rw_innovations = pm.Normal(
        "rw_innovations",
        dims=("president", "month_minus_origin"),
    )
    raw_rw = aet.cumsum(aet.concatenate([rw_init, rw_innovations], axis=-1), axis=-1)
    sd = pm.HalfNormal("shrinkage_pop", 0.2)
    month_president_effect = pm.Deterministic(
        "month_president_effect", raw_rw * sd, dims=("president", "month")
    )

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
            baseline
            + president_effect[president_id]
            + month_effect[month_id]
            + month_president_effect[president_id, month_id]
            + house_effect[pollster_by_method_id]
        ),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )

    idata = pm.sample(return_inferencedata=True)
```

```python
arviz.plot_trace(
    idata,
    var_names=["~popularity", "~rw"],
    filter_vars="regex",
);
```

That looks much better, doesn't it? Now we do see a difference in the different months, and the shrinkage standard deviation looks much more reasonable too, meaning that once we've accounted for the variation in popularity associated with the other effects, the different presidents' popularity isn't that different on a monthly basis -- i.e there _are_ cycles in popularity, no matter who the president is.

We could stop there, but, for fun, let's improve this model even further by:

1. Use a Beta-Binomial likelihood. We already saw in the completely pooled model that it improves fit and convergence a lot. Plus, it makes scientific sense: for a lot of reasons, each poll probably has a different true Binomial probability than all the other ones -- even when it comes from the same pollster; just think about measurement errors or the way the sample is different each time. Here, we parametrize the Beta-Binomial by its mean and precision, instead of the classical $\alpha$ and $\beta$ parameters. For more details about this distribution and parametrization, see [this blog post](https://alexandorra.github.io/pollsposition_blog/popularity/macron/gaussian%20processes/polls/2021/01/18/gp-popularity.html#Build-me-a-model).

2. Make sure that our different effects sum to zero. Think about the month effect. It only makes sense in a relative sense: some months are better than average, some other are worse, but you can't have _only_ good months -- they'd be good compared to what? So we want to make sure that the average month effect is 0, while allowing each month to be better or worse than average if needed. To do that, we use a Normal distribution whose dimensions are constrained to sum to zero. In PyMC, we can use the `ZeroSumNormal` distribution, that [Adrian Seyboldt](https://github.com/aseyboldt) contributed and kindly shared with us.

Ok, enough talking, let's code!

```python
from typing import *


def ZeroSumNormal(
    name: str,
    sigma: float = 1.0,
    *,
    dims: Union[str, Tuple[str]],
    model: Optional[pm.Model] = None,
):
    """
    Multivariate normal, such that sum(x, axis=-1) = 0.

    Parameters

    name: str
        String name representation of the PyMC variable.
    sigma: float, defaults to 1
        Scale for the Normal distribution. If none is provided, a standard Normal is used.
    dims: Union[str, Tuple[str]]
        Dimension names for the shape of the distribution.
        See https://docs.pymc.io/pymc-examples/examples/pymc3_howto/data_container.html for an example.
    model: Optional[pm.Model], defaults to None
        PyMC model instance. If ``None``, a model instance is created.

    Notes
    ----------
    Contributed by Adrian Seyboldt (@aseyboldt).
    """
    if isinstance(dims, str):
        dims = (dims,)

    model = pm.modelcontext(model)
    *dims_pre, dim = dims
    dim_trunc = f"{dim}_truncated_"
    (shape,) = model.shape_from_dims((dim,))
    assert shape >= 1

    model.add_coords({f"{dim}_truncated_": pd.RangeIndex(shape - 1)})
    raw = pm.Normal(
        f"{name}_truncated_", dims=tuple(dims_pre) + (dim_trunc,), sigma=sigma
    )
    Q = make_sum_zero_hh(shape)
    draws = aet.dot(raw, Q[:, 1:].T)

    return pm.Deterministic(name, draws, dims=dims)


def make_sum_zero_hh(N: int) -> np.ndarray:
    """
    Build a householder transformation matrix that maps e_1 to a vector of all 1s.
    """
    e_1 = np.zeros(N)
    e_1[0] = 1
    a = np.ones(N)
    a /= np.sqrt(a @ a)
    v = a + e_1
    v /= np.sqrt(v @ v)
    return np.eye(N) - 2 * np.outer(v, v)
```

```python
with pm.Model(coords=COORDS) as hierarchical_popularity:

    baseline = pm.Normal("baseline")
    president_effect = ZeroSumNormal("president_effect", sigma=0.15, dims="president")
    house_effect = ZeroSumNormal("house_effect", sigma=0.15, dims="pollster_by_method")
    month_effect = ZeroSumNormal("month_effect", sigma=0.15, dims="month")

    # need the cumsum parametrization to properly control the init of the GRW
    rw_init = aet.zeros(shape=(len(COORDS["president"]), 1))
    rw_innovations = pm.Normal(
        "rw_innovations",
        dims=("president", "month_minus_origin"),
    )
    raw_rw = aet.cumsum(aet.concatenate([rw_init, rw_innovations], axis=-1), axis=-1)
    sd = pm.HalfNormal("shrinkage_pop", 0.2)
    month_president_effect = pm.Deterministic(
        "month_president_effect", raw_rw * sd, dims=("president", "month")
    )

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
            baseline
            president_effect[president_id]
            + month_effect[month_id]
            + month_president_effect[president_id, month_id]
            + house_effect[pollster_by_method_id]
        ),
        dims="observation",
    )

    # overdispersion parameter
    theta = pm.Exponential("theta_offset", 1.0) + 10.0

    N_approve = pm.BetaBinomial(
        "N_approve",
        alpha=popularity * theta,
        beta=(1.0 - popularity) * theta,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )
#pm.model_to_graphviz(hierarchical_popularity)
```

```python
with hierarchical_popularity:
    idata = pm.sample(return_inferencedata=True)
```

```python
arviz.plot_trace(
    idata,
    var_names=["~popularity", "~truncated", "~rw_innovations"],
    filter_vars="regex",
    compact=True,
);
```

```python
arviz.summary(
    idata,
    round_to=2,
    var_names=["~popularity", "~truncated", "~rw_innovations"],
    filter_vars="regex",
)
```

```python
mean_house_effect = (
    idata.posterior["house_effect"].mean(("chain", "draw")).to_dataframe()
)
mean_house_effect.round(2)
```

```python
ax = mean_house_effect.plot.bar(figsize=(14, 8), rot=30)
ax.set_title("$>0$ bias means (pollster, method) overestimates the latent popularity");
```

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for ax, p in zip(axes.ravel(), idata.posterior.coords["president"]):
    post = idata.posterior.sel(president=p)
    post_pop = logistic(
        (
            #post["baseline"]
            post["president_effect"]
            + post["month_effect"]
            + post["month_president_effect"]
        ).stack(sample=("chain", "draw"))
    )
    post_pop = post_pop.isel(
        sample=np.random.choice(post_pop.coords["sample"].size, size=1000)
    )
    ax.plot(post.coords["month"], post_pop, alpha=0.01, color="blue", label=p)
    post_pop.median("sample").plot(
        ax=ax, color="orange", alpha=0.8, lw=2, label="Median"
    )
    ax.set_ylabel("Latent popularity")
    ax.set_xlabel("Months into term")
```

```python
with hierarchical_popularity:
    predictives = pm.sample_posterior_predictive(idata)
```

```python
data['p_approve_predicted'] = np.mean(predictives['N_approve'], axis=0) / data["samplesize"]
```

```python
predicted_approval_rates = data["p_approve_predicted"].values
approval_rates = data["p_approve"].values
newterm_dates = data.reset_index().groupby("president").first()["field_date"].values

dates = data.field_date

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(dates, predicted_approval_rates, "o", alpha=0.5, label="predicted")
ax.plot(dates, approval_rates, "o", alpha=0.5, label="observed")
ax.set_ylim(0, 1)
ax.set_ylabel("Does approve")
for date in newterm_dates:
    ax.axvline(date)
fig.legend();
```

## TODO

- Posterior predictive analysis: distribution of $p_{\mathrm{approve}}$ for each pollster and method. We can plot the approval rates for each poll for each president.

- Re-read the paper by Gellman et al. on predicting the US presidential election. We may be able to catch something new given our experience with this first model.

- Estimate `month_effect` with a GRW too? This could be easier for the model to first estimate the temporal dependency only for month, and then do that for each month and president.

- Try a GP for the temporal dependency? This would estimate the _covariation_ between months for free, and also GPs tend to behave better than GRW.

- We do not include time correlations in the model, but it is obvious that there is a *dynamic* in the popularity and the popularity at time $t$ also depends on the popularity at times before the previous months; we could add time correlations.

```python
%load_ext watermark
%watermark -n -u -v -iv
```
