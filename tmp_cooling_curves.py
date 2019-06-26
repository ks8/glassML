import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.loadtxt('N2e7_cooling_curve.txt')

x = np.linspace(1.99025, 0.05, 200)

plt.plot(x, data[:, 1], color='b', label=r'$t_{cool} = 2 \times 10^7$')
plt.axvline(0.37, color='k', linestyle=':')
plt.scatter(1.99205, -3.7647, color='r', marker='P', s=80)
plt.text(1.99205, -3.773, '1', color='k')

plt.scatter(1.95125, -3.7662, color='darkorange', marker='P', s=80)
plt.text(1.95125, -3.7745, '2', color='k')

plt.scatter(1.9025, -3.7676, color='mediumturquoise', marker='P', s=80)
plt.text(1.9025, -3.7759, '3', color='k')

plt.scatter(1.85375, -3.7689, color='darkviolet', marker='P', s=80)
plt.text(1.85375, -3.7772, '4', color='k')

plt.scatter(1.75625, -3.7719, color='magenta', marker='P', s=80)
plt.text(1.75625, -3.7802, '5', color='k')

plt.scatter(0.635, -3.8511, color='gold', marker='P', s=80)
plt.text(0.635, -3.8594, '6', color='k')

plt.scatter(0.48875, -3.8784, color='lime', marker='P', s=80)
plt.text(0.48875, -3.8867, '7', color='k')

plt.scatter(0.05, -3.9029, color='g', marker='s', s=80)
plt.ylabel('Inherent Structure Energy Per Particle (LJ units)')
plt.xlabel('Temperature (LJ units)')
plt.legend()

plt.savefig('N2e7_cooling_curve_plot')

# data1 = np.loadtxt('N2e5_cooling_curve.txt')
# data2 = np.loadtxt('N2e7_cooling_curve.txt')
# x = np.linspace(1.99205, 0.05, 200)
#
# plt.plot(x[80:], data1[80:, 1], color='r', label=r'$t_{cool} = 2 \times 10^5$')
# plt.plot(x[80:], data2[80:, 1], color='b', label=r'$t_{cool} = 2 \times 10^7$')
# plt.scatter(0.54725, -3.86698, color='deeppink', marker='P', s=80)
# plt.text(0.54725, -3.873, '8', color='k')
# plt.scatter(0.05, -3.86744, color='g', marker='s', s=80)
# # plt.hlines(-3.86744, 0.05, 0.54, linestyle='dashed', color='lightgrey')
# plt.plot([0.05, 0.42], [-3.9029, -3.90076], color='k', linewidth=1, linestyle='dotted')
# plt.plot([0.35, 0.557], [-3.90477, -3.865], color='k', linewidth=1, linestyle='dotted')
# plt.plot([0.35, 0.7032], [-3.8689, -3.8323], color='k', linewidth=1, linestyle='dotted')
# plt.plot([0.05, 0.44], [-3.8674, -3.8646], color='k', linewidth=1, linestyle='dotted')
# plt.legend()
#
# plt.ylabel('Inherent Structure Energy Per Particle (LJ units)')
# plt.xlabel('Temperature (LJ units)')
#
# plt.savefig('comparing_cooling_curves')