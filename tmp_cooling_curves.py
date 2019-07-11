import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# data = np.loadtxt('N2e7_cooling_curve.txt')
#
# x = np.linspace(1.99025, 0.05, 200)
#
# plt.plot(x, data[:, 1], color='g', label=r'$t_{cool} = 2 \times 10^7$', zorder=1)
# plt.axvline(0.37, color='k', linestyle=':', zorder=2)
# plt.scatter(1.99205, -3.7647, color='blue', marker='o', s=80, zorder=3, label='liquid')
# plt.text(1.99205, -3.775, '1', color='k', weight='bold', fontsize='12')
#
# # plt.scatter(1.95125, -3.7662, color='darkorange', marker='P', s=80)
# # plt.text(1.95125, -3.7745, '2', color='k')
# #
# # plt.scatter(1.9025, -3.7676, color='mediumturquoise', marker='P', s=80)
# # plt.text(1.9025, -3.7759, '3', color='k')
#
# # plt.scatter(1.85375, -3.7689, color='darkviolet', marker='P', s=80)
# # plt.text(1.85375, -3.7772, '4', color='k')
#
# plt.scatter(1.75625, -3.7719, color='blue', marker='o', s=80, zorder=3)
# plt.text(1.75625, -3.7808, '2', color='k', weight='bold', fontsize='12')
#
# plt.scatter(0.95675, -3.8118, color='blue', marker='o', s=80, zorder=3)
# plt.text(0.95675, -3.8205, '3', color='k', weight='bold', fontsize='12')
#
# # plt.scatter(0.635, -3.8511, color='gold', marker='P', s=80)
# # plt.text(0.635, -3.8594, '6', color='k')
#
# plt.scatter(0.54765, -3.8655, color='blue', marker='o', s=80, zorder=3)
# plt.text(0.54765, -3.875, '4', color='k', weight='bold', fontsize='12')
#
# plt.scatter(0.44, -3.8863, color='blue', marker='o', s=80, zorder=3)
# plt.text(0.44, -3.896, '5', color='k', weight='bold', fontsize='12')
#
# # plt.scatter(0.48875, -3.8784, color='blue', marker='o', s=80)
# # plt.text(0.48875, -3.8867, '7', color='k')
#
# plt.scatter(0.05, -3.9029, color='m', marker='o', s=80, zorder=3, label='glass')
# plt.text(0.08, -3.91, '1-5', color='k', weight='bold', fontsize='12')
# plt.ylabel('Inherent Structure Energy Per Particle (LJ units)')
# plt.xlabel('Temperature (LJ units)')
# plt.legend()
#
# plt.savefig('N2e7_cooling_curve_plot')

data1 = np.loadtxt('N2e5_cooling_curve.txt')
data2 = np.loadtxt('N2e7_cooling_curve.txt')
x = np.linspace(1.99205, 0.05, 200)

plt.plot(x[80:], data1[80:, 1], color='slategrey', label=r'$t_{cool} = 2 \times 10^5$', zorder=1)
plt.plot(x[80:], data2[80:, 1], color='g', label=r'$t_{cool} = 2 \times 10^7$', zorder=1)
plt.scatter(0.54725, -3.86698, color='b', marker='o', s=80, zorder=3, label='liquid')
plt.text(0.54725, -3.875, '6', color='k', weight='bold', fontsize='12')
plt.scatter(0.05, -3.86744, color='m', marker='o', s=80, zorder=3, label='glass')
plt.text(0.05, -3.875, '6', color='k', weight='bold', fontsize='12')
# plt.hlines(-3.86744, 0.05, 0.54, linestyle='dashed', color='lightgrey')
plt.plot([0.05, 0.42], [-3.9029, -3.90076], color='k', linewidth=1, linestyle='dotted', zorder=2)
plt.plot([0.35, 0.557], [-3.90477, -3.865], color='k', linewidth=1, linestyle='dotted', zorder=2)
plt.plot([0.35, 0.7032], [-3.8689, -3.8323], color='k', linewidth=1, linestyle='dotted', zorder=2)
plt.plot([0.05, 0.44], [-3.8674, -3.8646], color='k', linewidth=1, linestyle='dotted', zorder=2)
plt.legend()

plt.ylabel('Inherent Structure Energy Per Particle (LJ units)')
plt.xlabel('Temperature (LJ units)')

plt.savefig('comparing_cooling_curves')