import numpy as np
import matplotlib.pyplot as plt

# load the data
x = [1, 2, 3]
plt.rcParams['font.size'] = 18


# name = 'box_inpainting'
# base = [0.7660937500000001]
# std_base = [0.07398832651802242]

# best_of_n = [0.717296875, 0.6844687499999996, 0.6611093750000001]
# std_best_of_n = [0.07505158219674234, 0.07846794105516915, 0.08063926563473513]

# search = [0.6917499999999998, 0.6246562499999998, 0.5875625]
# std_search = [0.08314727145252572, 0.08918331170088663, 0.10425507706462069]

# group_search = [0.6926562500000001, 0.6090000000000001, 0.542609375]
# std_group_search = [0.08255740781987707, 0.07155963422209478, 0.08475121997416542]



# name = 'gaussian_blur'
# base = [1.02590625]
# std_base = [0.06414624861157119]

# best_of_n = [0.994359375, 0.9693437499999997, 0.9505000000000001]
# std_best_of_n = [0.06792145445004379, 0.07060148253356653, 0.07540743497825661]

# search = [0.9743125000000001, 0.9287343749999993, 0.8934687500000001]
# std_search = [0.0677624239069855, 0.058967481024369474, 0.07076721715199417]

# group_search = [0.9775468750000001, 0.9056562500000003, 0.8487187499999996]
# std_group_search = [0.06427672831386468, 0.061126717447753574, 0.07447429689790631]


name = 'super_resolution_x6'
base = [1.036968253968254]
std_base = [0.07750943992768723]

best_of_n = [1.005375, 0.9803281250000002, 0.9643750000000001]
std_best_of_n = [0.07360037788625816, 0.07630429515423345, 0.08648509625941339]

search = [0.9815468749999999, 0.9231875000000002, 0.8782343749999999]
std_search = [0.07752095073420073, 0.08110465365039171, 0.08545937305737374]

group_search = [0.9715156250000001, 0.8945624999999998, 0.8337968750000001]  # tofill
std_group_search = [0.07695310913705422, 0.08160860765721958, 0.09153537220787586] # tofill


plt.plot([0] + x, base + best_of_n, label='BestOfN', color='firebrick', marker='o')
plt.errorbar(x, best_of_n, yerr=std_best_of_n, color='firebrick', alpha=0.25)

plt.plot([0] + x, base + search, label='GlobalSearch', color='steelblue', marker='o')
plt.errorbar(x, search, yerr=std_search, color='steelblue', alpha=0.25)

plt.plot([0] + x, base + group_search, label='GroupSearch', color='forestgreen', marker='o')
plt.errorbar(x, group_search, yerr=std_group_search, color='forestgreen', alpha=0.25)

plt.plot([0], base, label='MPGD', color='black', marker='o')
plt.errorbar([0], base, yerr=std_base, color='black', alpha=0.25)

plt.xlabel('Number of Particles (N)')
plt.ylabel('FaceSimilarity (\u2193)')
# plt.ylim([0.4, 1])
# plt.grid()
plt.xticks(ticks=[0, 1, 2, 3], labels=['1', '2', '4', '8'])
plt.tick_params(axis='both')
plt.legend()
plt.tight_layout()

# make the figure transparent
plt.savefig(f'{name}_face_similarity_vs_n.pdf', bbox_inches='tight', dpi=300, transparent=True)

print('done')