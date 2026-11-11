import glob, os, json, numpy as np, cv2, scipy.io as sio
from skimage.filters import sato
from skimage.morphology import skeletonize
from sklearn.model_selection import StratifiedGroupKFold
from common import *
ROOT = "../data/cfd/CrackForest-dataset-master"; P = 64
X, y, grp = [], [], []
for gt in sorted(glob.glob(f"{ROOT}/groundTruth/*.mat")):
    id_ = os.path.basename(gt)[:-4]; im = cv2.imread(f"{ROOT}/image/{id_}.jpg")
    seg = sio.loadmat(gt, squeeze_me=True, struct_as_record=False)["groundTruth"].Segmentation == 2
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); g = cv2.medianBlur(g, 5)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    can = cv2.Canny(g, 40, 120) > 0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1); mag = np.hypot(gx, gy)
    rid = sato(g.astype(np.float32) / 255, sigmas=[1.5, 2.5], black_ridges=True)
    loc = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8) > 0
    H, W = g.shape
    for yy in range(0, H - P + 1, P):
        for xx in range(0, W - P + 1, P):
            s = seg[yy:yy+P, xx:xx+P].sum(); 
            if 0 < s < 30: continue      # patch ambigu dibuang
            sl = (slice(yy, yy+P), slice(xx, xx+P)); gp = g[sl]
            sk = skeletonize(loc[sl]).sum() / P
            edge = [can[sl].mean(), mag[sl].mean(), mag[sl].std(), np.percentile(mag[sl], 95)]
            ridge = [rid[sl].mean(), rid[sl].max(), np.percentile(rid[sl], 95), sk, loc[sl].mean()]
            tex = np.concatenate([glcm_feats(gp), lbp_hist(gp), [gp.mean(), gp.std()]])
            X.append(np.concatenate([edge, ridge, tex])); y.append(int(s >= 30)); grp.append(int(id_))
X = np.array(X); y = np.array(y); grp = np.array(grp)
np.savez("../results/crack_feats.npz", X=X, y=y, grp=grp)
iE, iR, iT = slice(0, 4), slice(4, 9), slice(9, X.shape[1])
sets = {"Tepi (Canny/Sobel)": X[:, iE], "Tepi+ridge+morfologi": X[:, 0:9], "Tekstur (GLCM+LBP)": X[:, iT], "Semua": X}
def split(seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, grp))
out = {k: evaluate(v, y, grp, split, ["Non-retak", "Retak"]) for k, v in sets.items()}
# cek: tak ada citra di train dan test sekaligus
for tr, te in split(0): assert not set(grp[tr]) & set(grp[te])
# pembanding bocor (random patch split) hanya untuk diskusi
from sklearn.model_selection import StratifiedKFold
leak = evaluate(X, y, grp, lambda s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)), ["Non-retak", "Retak"])
json.dump(dict(classes=["Non-retak", "Retak"], n=len(y), counts=np.bincount(y).tolist(), n_img=len(set(grp)), res=out, leak_random_split=leak), open("../results/crack.json", "w"), indent=1)
print("n", len(y), np.bincount(y))
for k, r in out.items(): print("CRACK", k, {m: f"{v['mean']:.3f}±{v['std']:.3f}" for m, v in r.items()})
print("CRACK leak(random patch split, Semua)", {m: f"{v['mean']:.3f}±{v['std']:.3f}" for m, v in leak.items()})
