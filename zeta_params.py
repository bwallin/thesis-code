from scipy import array

dx = .7

mu_g_0 = -25
sigma_g_0 = 1
mu_h_0 = 30
sigma_h_0 = 1

phi = .95
mu_h = 30
sigma_g = .1
sigma_z_g = .2
sigma_h = 2
sigma_z_h = 4

noise_proportion = .3
canopy_cover = array([0, .5, 1])

cover_transition_matrix = array([[.98, .017, .003],
                                  [.01, .98, .01],
                                  [.001, .004, .995]])
