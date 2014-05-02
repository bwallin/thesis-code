from scipy import array

dx = .7

mu_g_0 = -25
sigma_g_0 = 1
mu_h_0 = 30
sigma_h_0 = 1

phi = .95
mu_h = 30
transition_var_g = .1
observation_var_g = .2
transition_var_h = 2
observation_var_h = 4

noise_proportion = .3
canopy_cover = array([0, .25, .5, .75])

cover_transition_matrix = array([[.99, .0033, .0033, .0033],
                                 [.0033, .99, .0033, .0033],
                                 [.0033, .0033, .99, .0033],
                                 [.00033, .00033, .00033, .999]])
