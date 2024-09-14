import math
import sys

import matplotlib
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({
  "text.usetex": True,
  "text.latex.preamble": r"\usepackage{biolinum} \usepackage{libertineRoman} \usepackage{libertineMono} \usepackage{biolinum} \usepackage[libertine]{newtxmath}",
  'ps.usedistiller': "xpdf",
})


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

inp_dir = sys.argv[1]
a_idx = int(sys.argv[2])
use_div = int(sys.argv[3])

ab = np.loadtxt(f"{inp_dir}/output_div_{a_idx:0>3}/div_vs_ab.txt")
#ab = np.loadtxt(f"{inp_dir}/output_{a_idx:0>3}/div_vs_ab.txt")
n_steps = ab.shape[0]
print(ab.min(), ab.max())
ba = np.loadtxt(f"{inp_dir}/output_div_{a_idx:0>3}/div_vs_ba.txt")
#ba = np.loadtxt(f"{inp_dir}/output_{a_idx:0>3}/div_vs_ba.txt")
print(ba.min(), ba.max())

ios_ab = np.loadtxt(f"../inbetweening/{inp_dir}/int_{a_idx:0>3}/fwd/ios_dirichlet_N_N.txt")
print(ios_ab)
print(ios_ab.min(), ios_ab.max())
ios_ba = np.loadtxt(f"../inbetweening/{inp_dir}/int_{a_idx:0>3}/bwd/ios_dirichlet_N_N.txt")
ios_ba = ios_ba[::-1]
print(ios_ba)
print(ios_ba.min(), ios_ba.max())

ios = np.linspace(ios_ab, ios_ba, n_steps)
ios = np.transpose(ios, (1, 0))

#W = ab - ba
#W = (W - W.min()) / (W.max() - W.min())
#W *= (1 - ios / ios.max())
#W = ab ** 2 + ba ** 2 + 1e-4 * ios
#W = ab - ba + 1e-1 * ios
div = ab + ba
div = np.tile(div[:, None], (1, len(div)))
div = np.cumsum(div, axis=1) # [N, ]
#div = np.tile(div[:, None], (1, len(div)))
#print(div)

W = div + ios # [N, ] + [N, N] = [1, N] + [N, N]
#W = ios

G = nx.DiGraph()
_as = []
_bs = []
_cs = []
if n_steps >= 24:
    n_neigh = n_steps // 8
else:
    n_neigh = 3
W = np.full((n_steps, n_steps), fill_value=0, dtype="float64")
#_Iou = np.full((n_steps, n_steps), fill_value=0, dtype="float64")
#_Div = np.full((n_steps, n_steps), fill_value=0, dtype="float64")
#_Dist = np.full((n_steps, n_steps), fill_value=0, dtype="float64")
for i in range(n_steps - 1):
    for j in range(n_steps if i != 0 else 1):
        I = i * n_steps + j
        G.add_node(I, pos=(i, j))

        #for k in range(j, min(j + 4, n_steps)):
        for k in range(j if i + 1 != n_steps - 1 else n_steps - 1, min(j + n_neigh, n_steps)):#n_steps):
            J = (i + 1) * n_steps + k
            G.add_node(J, pos=(i + 1, k))

            #weight = ab[i + 1, k] ** 2 + ba[i + 1, k] ** 2
            #weight = ab[i + 1, k] - ba[i + 1, k]
            #weight = ios[i + 1, k]
            #weight = W[i + 1, k] - W[i, j]
            _a = (div[i, k] + div[i + 1, k]) / 2
            _as.append(_a)
            #_Div[i + 1, k] += _a

            _b = (ios[i + 1, k] + ios[i, j])
            _bs.append(_b)
            #_Iou[i + 1, k] += _b

            _c = math.sqrt(
                (
                    (1 / (n_steps - 1)) ** 2
                    +
                    ((k - j)/ (n_steps - 1)) ** 2
                 )
            )
            #_Dist[i + 1, k] += _c
            _cs.append(_c)

            weight = 0
            if use_div > 0:
                weight += _a
            #weight += 100 * _b
            weight += 175 * _b
            #weight += 200 * _b  # sneg
            #weight += 300 * _b  # aladin
            weight += 5 * _c

            W[i + 1, k] += weight
            assert not G.has_edge(I, J)
            G.add_edge(I, J, weight=weight)

src = 0
tgt = n_steps * n_steps - 1
#path = nx.dijkstra_path(G, src, tgt)
path = nx.bellman_ford_path(G, src, tgt)
#print(path)
print("sum path", sum(G[u][v]["weight"] for u, v in zip(path, path[1:])))
js = "\n".join(
    f"img_deformed_dirichlet_N_N_{i:0>3}_{p % n_steps:0>3}.png"
    for i, p in enumerate(path)
)
with open("anim.txt", "w") as f:
    f.write(js)

ds = "\n".join(
    f"img_deformed_dirichlet_N_N_{i:0>3}_{i:0>3}.png"
    for i, _ in enumerate(path)
)
with open("diag.txt", "w") as f:
    f.write(ds)

with open(f"{inp_dir}/output_{a_idx:0>3}/diag.txt", "w") as f:
    f.write(ds)

with open(f"{inp_dir}/output_{a_idx:0>3}/anim.txt", "w") as f:
    f.write(js)
#with open(f"{inp_dir}/output_{a_idx:0>3}/diag.txt", "w") as f:
    #f.write(js)


print("_as", min(_as), np.mean(_as), max(_as))
print("_bs", min(_bs), np.mean(_bs), max(_bs))
print("_cs", min(_cs), np.mean(_cs), max(_cs))
print((np.array(list(zip(_as, _bs, _cs))).mean(0)))

#plt.bar(np.arange(1, 4), (np.array(list(zip(_as, _bs, _cs))).mean(0)), width=0.1)
#plt.show()
#plt.close()
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Iou")
plt.imshow(ios)
#plt.imshow(_Iou)
#plt.colorbar(plt.pcolor(ios[::-1]))
#plt.imshow(ba)
#pos = nx.get_node_attributes(G, "pos")
#nx.draw(G, pos, with_labels=True)
#labels = nx.get_edge_attributes(G, "weight")
#nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.subplot(2, 2, 2)
plt.title("Div")
plt.imshow(div)
#plt.imshow(_Div)
#plt.colorbar(plt.pcolor(div[::-1]))

plt.subplot(2, 2, 3)
#print((ab + ba))
_hm = np.tile((ab + ba)[:, None], (1, len(ba)))
#plt.title("Dist")
plt.imshow(_hm)
#plt.imshow(_Dist)
#print(hm)
#hm = plt.pcolor(_hm[::-1])
#for i in range(_hm.shape[0]):
#    for j in range(_hm.shape[1]):
#        plt.text(
#            j + 0.5,
#            n_steps - 1 - i + 0.5,
#            "%d %d" % (i, j),
#            fontsize=4.0,
#            horizontalalignment="center",
#            verticalalignment="center",
#        )
#plt.colorbar(hm)

plt.subplot(2, 2, 4)
plt.imshow(W)
#plt.colorbar(plt.pcolor(W[::-1]))

i = 0
#for path in nx.all_shortest_paths(G, src, tgt, weight="weight", method="bellman-ford"):
for path in [path]:
    for p in path:
        i, j = p // n_steps, p % n_steps
        #plt.scatter(j + 0.5, n_steps - 1 - i + 0.5, c="red")
        plt.scatter(j, i, c="red")

plt.tight_layout()
plt.savefig(f"{inp_dir}/path_{a_idx}.svg")
#plt.show()
plt.close()

#plt.figure(figsize=(10, 10))
#pos = nx.get_node_attributes(G, "pos")
#nx.draw(G, pos, style="dashed")
#labels = nx.get_edge_attributes(G, "weight")
#nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#plt.show()
#plt.close()

#plt.imshow(nx.adjacency_matrix(G).todense())
#plt.show()
#plt.close()


plt.figure(figsize=(6, 6))
#plt.plot([0, len(path) - 1], [0, len(path) - 1], ls=".", color="red")
plt.scatter(range(len(path)), range(len(path)), c="red")
for p in path:
    i, j = p // n_steps, p % n_steps
    plt.scatter(j, i, c="green")

plt.xlabel(r"frame interpolation step $\alpha$")
ticks = list(range(0, len(path) + 1, 2))
if n_steps >= 24:
    labels = [
        "0",
        "", "", "", "", "", "0.5", "", "", "", "", "",
        "1",
    ]
else:
    labels = [
        "0",
        "", "0.5", "",
        "1",
    ]

try:
    plt.xticks(ticks, labels=labels)
except:
    pass

plt.ylabel(r"time $t$")
try:
    plt.yticks(ticks, labels=labels)
except:
    pass
plt.grid(ls="--")
plt.axis("equal")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig(f"{inp_dir}/path_diag_{a_idx}.svg")
plt.show()
plt.close()
