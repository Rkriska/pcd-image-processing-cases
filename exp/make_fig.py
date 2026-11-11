import json, glob, numpy as np, cv2, scipy.io as sio, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 6.5, "font.family": "serif"})
R = "../results/"
pal = json.load(open(R+"palm.json")); ric = json.load(open(R+"rice.json")); cra = json.load(open(R+"crack.json"))
for n, d in [("palm", pal), ("rice", ric), ("crack", cra)]:
    k = list(d["res"])[-1] if n != "crack" else "Semua"
    for m in ["SVM-RBF", "RF"]:
        print(n, k, m, np.array(d["res"][k][m]["cm"]).tolist())
fig = plt.figure(figsize=(7.2, 1.55))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.08, 2.55], wspace=0.2)
def show(ax, im, t):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(t, fontsize=6.5, pad=2); ax.axis("off")
pf = json.load(open(R+"palm_files.json")); masks = np.load(R+"palm_masks.npy")
i = [j for j, f in enumerate(pf) if "/Ripe/" in f][3]
im = cv2.resize(cv2.imread(pf[i]), (256, 256)); cnt, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, cnt, -1, (255, 255, 0), 2); show(fig.add_subplot(gs[0]), im, "(a) TBS matang+mask")
im = cv2.imread("../data/rice/Brown spot/DSC_0100.jpg"); im = im[:, 150:500]
show(fig.add_subplot(gs[1]), cv2.resize(im, (320, int(320*im.shape[0]/im.shape[1]))), "(b) padi: brown spot")
R0 = "../data/cfd/CrackForest-dataset-master"; im = cv2.imread(f"{R0}/image/001.jpg")
seg = sio.loadmat(f"{R0}/groundTruth/001.mat", squeeze_me=True, struct_as_record=False)["groundTruth"].Segmentation == 2
im[seg] = (0, 0, 255); show(fig.add_subplot(gs[2]), im, "(c) retak+anotasi")
ax = fig.add_subplot(gs[4])
lab = ["Sawit\nwarna", "Sawit\nw+t", "Padi\nlesi", "Padi\nsemua", "Retak\ntepi", "Retak\nsemua"]
get = lambda d, k, m: (d["res"][k][m]["mean"], d["res"][k][m]["std"])
ks = [(pal, "Warna (mask)"), (pal, "Warna+tekstur"), (ric, "Warna lesi"), (ric, "Warna lesi+bentuk+tekstur"), (cra, "Tepi (Canny/Sobel)"), (cra, "Semua")]
x = np.arange(len(ks)); w = 0.38
for o, (m, c) in enumerate([("SVM-RBF", "0.25"), ("RF", "0.7")]):
    v = [get(d, k, m) for d, k in ks]
    ax.bar(x + (o-.5)*w, [a for a, _ in v], w, yerr=[b for _, b in v], color=c, label=m, capsize=1.5, error_kw={"lw": .6})
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=5.5); ax.set_ylim(0.5, 1.0); ax.set_ylabel("macro-F1"); ax.legend(fontsize=6, loc="upper left", ncol=2, frameon=False)
ax.set_title("(d) Ablasi fitur (mean ± std antar lipatan)", fontsize=6.5, pad=2)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.savefig(R+"fig_overview.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(R+"fig_overview_v2.png", dpi=200, bbox_inches="tight")
