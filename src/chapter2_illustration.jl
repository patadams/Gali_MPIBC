#=
Implement the RBC model described in sections 2.1-2.3 of:
"Monetary Policy, Inflation, and the Business Cycle" by Jordi Gali,
using a stationary ARMA process for log productivity.

Author: Pat Adams (patrick.augustine.adams@gmail.com)
=#

using QuantEcon
using Plots

doc"""
Represents an RBC model.

Includes preference and technology parameters, as well as
a stochastic process for log productivity ``a_t = \log A_t``,
so that equilibrium paths and impulse responses for all real
variables can be computed.

##### Fields
- `sigma::Real` : Preference parameter for consumption
- `phi::Real` : Preference parameter for labor
- `rho::Real` : Rate of time preference
- `alpha::Real` : Parameter for production function
- `a_unc_mean::Real` : Unconditional mean for log productivity 
- `a_arma::ARMA` : ARMA process for log productivity
"""
type RBC_model
    sigma::Real
    phi::Real
    rho::Real
    alpha::Real
    a_unc_mean::Real
    a_arma::ARMA
end


doc"""
Compute the "reduced form" parameters linking real output, employment,
and real wage to productivity in the RBC model:

```math
          y_t = \psi_{ya} * a_t + \theta_y
          n_t = \psi_{na} * a_t + \theta_n
    w_t - p_t = \psi_{wa} * a_t + \theta_w
```

See section 2.3 of the textbook for more details.

##### Arguments
- `model::RBC_model` : Instance of `RBC_model` type.

##### Returns
- `psi_ya::Real`, `theta_y::Real` : Reduced form parameters for real output
- `psi_na::Real`, `theta_n::Real` : Reduced form parameters for employment
- `psi_wa::Real`, `theta_w::Real` : Reduced form parameters for real wage
"""
function compute_RBC_reduced_form(model::RBC_model)
    # Unpack model parameters.
    sigma, phi, rho, alpha = model.sigma, model.phi, model.rho, model.alpha
    
    # Construct reduced form parameters.
    denom = sigma * (1 - alpha) + phi + alpha # for convenience
    psi_na = (1 - sigma) / denom
    theta_n = log(1 - alpha) / denom
    psi_ya = (1 + phi) / denom
    theta_y = (1 - alpha) * theta_n
    psi_wa = (sigma + phi) / denom
    theta_w = (sigma * (1 - alpha) + phi) * log(1 - alpha) / denom
    
    return psi_ya, theta_y, psi_na, theta_n, psi_wa, theta_w
end



doc"""
Simulate equilibrium paths for RBC model starting from steady state.

##### Arguments
- `model::RBC_model` : Instance of `RBC_model` type.
- `;ts_length::Integer(90)` : Length of simulation
- `;impulse_length::Integer(30)` : Horizon for calculating impulse response
  for MA(\infty) representation of ARMA process for log productivity
  (see method `impulse_response` from QuantEcon file `arma.jl` for more information)

##### Returns
- `a_sim::Array{Float64,1}` : Simulated path for productivity
- `y_sim::Array{Float64,1}` : Simulated path for real output
- `n_sim::Array{Float64,1}` : Simulated path for employment
- `r_sim::Array{Float64,1}` : Simulated path for real interest rate
- `w_sim::Array{Float64,1}` : Simulated path for real wage
"""
function simulate_RBC(model::RBC_model; ts_length=90, impulse_length=30)
    J = impulse_length
    T = ts_length
    
    # Unpack model parameters and stochastic process for a_t.
    sigma, phi, rho, alpha = model.sigma, model.phi, model.rho, model.alpha
    a_unc_mean, a_arma = model.a_unc_mean, model.a_arma

    # Compute reduced form parameters.
    psi_ya, theta_y, psi_na, theta_n, psi_wa, theta_w = compute_RBC_reduced_form(model)

    # Simulate path for productivity.
    # a_t and E_t(a_{t+1}) are computed from the MA(\infty) representation of a_arma.
    # Adapted from method `simulation` from QuantEcon file `arma.jl`.
    a_impulse_response = impulse_response(a_arma; impulse_length=impulse_length)
    epsilon = [zeros(J); a_arma.sigma * randn(T)]
    a_sim = Array{Float64}(T) # a_sim[t] = a_t
    a_cond_mean = Array{Float64}(T) # a_cond_mean[t] = E_t(a_{t+1})
    for t=1:T
        a_sim[t] = a_unc_mean + dot(epsilon[t:J+t-1], reverse(a_impulse_response))
        a_cond_mean[t] = a_unc_mean + dot(epsilon[t:J+t-2], reverse(a_impulse_response[2:J]))
    end
    
    # Compute path for real variables given simulated path for productivity.
    y_sim = psi_ya * a_sim + theta_y    
    n_sim = psi_na * a_sim + theta_n
    r_sim = rho + sigma * psi_ya * (a_cond_mean - a_sim)    
    w_sim = psi_wa * a_sim + theta_w

    return a_sim, y_sim, n_sim, r_sim, w_sim
end



### Example: simulate equilibrium paths and create plots
sigma = 2
phi = 0
rho = 0.05
alpha = 0.3
a_unc_mean = 0
a_arma = ARMA([0.75]) # persistent AR(1) process for log productivity
model = RBC_model(sigma, phi, rho, alpha, a_unc_mean, a_arma)

T = 100
a_sim, y_sim, n_sim, r_sim, w_sim = simulate_RBC(model, ts_length=T)
p = plot(1:T, [a_sim y_sim n_sim r_sim], color=["blue" "orange" "red" "green"], labels=["a_t" "y_t" "n_t" "r_t"], layout=(4,1))
p.subplots[1].attr[:title] = "sigma=$sigma, phi=$phi, rho=$rho, alpha=$alpha"
plot!()
