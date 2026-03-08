# experiments.py

EXPERIMENTS = [

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=10,
temp_sample=0.1,
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=20.0,
w_constraint=1000,
sigma_min=0.02,
noise_scale=0.3,
w_u0=0.1
),

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=50, #############
temp_sample=0.1,
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=20.0,
w_constraint=1000,
sigma_min=0.02,
noise_scale=0.3,
w_u0=0.1
),

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=10,
temp_sample=0.1,
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=20.0,
w_constraint=1500, #############
sigma_min=0.02,
noise_scale=0.3,
w_u0=0.1
),

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=10,
temp_sample=0.1,
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=30.0, #############
w_constraint=1000,
sigma_min=0.02,
noise_scale=0.3,
w_u0=0.1
),

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=10,
temp_sample=0.1,
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=10.0, #############
w_constraint=1000,
sigma_min=0.02,
noise_scale=0.3,
w_u0=0.1
),

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=10,
temp_sample=0.1,
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=10.0,
w_constraint=1000,
sigma_min=0.02,
noise_scale=0.1, #############
w_u0=0.1
),

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=10,
temp_sample=0.1,
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=20.0,
w_constraint=1000,
sigma_min=0.02,
noise_scale=0.2, #############
w_u0=0.1
),

dict(
Nsample=1024,
Hsample=50,
Ndiffuse=10,
temp_sample=0.01, #############
beta0=0.0005,
betaT=0.01,
w_tracking=10.0,
w_smooth=20.0,
w_constraint=1000,
sigma_min=0.02,
noise_scale=0.3,
w_u0=0.1
)


]