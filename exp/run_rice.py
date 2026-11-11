import glob, os, json, numpy as np, cv2
from skimage.measure import label, regionprops
from sklearn.model_selection import StratifiedKFold
from common import *
CL = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]
def leaf_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = np.concatenate([lab[:5].reshape(-1,3), lab[-5:].reshape(-1,3)])
    d = np.linalg.norm(lab - np.median(b, 0), axis=2)
    d8 = np.clip(d / max(d.max(),1) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(cv2.GaussianBlur(d8, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8)) > 0
    return cv2.erode(m.astype(np.uint8), np.ones((5,5),np.uint8)) > 0   # buang tepi/halo
def feats(f):
    im = cv2.imread(f); h, w = im.shape[:2]; s = 1024 / max(h, w)
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lm = leaf_mask(im); hsv, lab = hsv_lab(im)
    # lesi = piksel daun yang tidak hijau (a* di atas ambang) -> Otsu pada a* di dalam daun
    a = lab[..., 1]; ar = a[lm]
    thr = max(-8.0, np.percentile(ar, 50) + 8)
    les = lm & (a > thr); les = cv2.morphologyEx(les.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3),np.uint8)) > 0
    if les.sum() < 30:
        les = lm & (lab[...,0] <= np.percentile(lab[...,0][lm], 5))
    lc = label(les); rp = [r for r in regionprops(lc) if r.area >= 15] or regionprops(lc)
    A = np.array([r.area for r in rp], float)
    ecc = np.average([r.eccentricity for r in rp], weights=A); sol = np.average([r.solidity for r in rp], weights=A)
    ext = np.average([r.extent for r in rp], weights=A)
    ar_ = np.average([r.major_axis_length / max(r.minor_axis_length, 1) for r in rp], weights=A)
    shape = np.array([les.sum() / lm.sum(), len(rp) / (lm.sum() / 1e4), A.mean() / lm.sum(), ecc, sol, ext, ar_])
    leafc = np.concatenate([moments(hsv[lm]), moments(lab[lm])])
    lesc = np.concatenate([moments(hsv[les]), moments(lab[les])])
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); ys, xs = np.where(lm)
    gg = g[ys.min():ys.max()+1, xs.min():xs.max()+1].copy(); gg[~lm[ys.min():ys.max()+1, xs.min():xs.max()+1]] = 0
    tex = np.concatenate([glcm_feats(gg), lbp_hist(gg)])
    return leafc, lesc, shape, tex, lm, les, im
D = {k: [] for k in ["leaf", "les", "shape", "tex"]}; y = []; files = []
for ci, c in enumerate(CL):
    for f in sorted(glob.glob(f"../data/rice/{c}/*")):
        r = feats(f)
        for k, v in zip(D, r[:4]): D[k].append(v)
        y.append(ci); files.append(f)
y = np.array(y); D = {k: np.array(v) for k, v in D.items()}
np.savez("../results/rice_feats.npz", y=y, **D); json.dump(files, open("../results/rice_files.json","w"))
sets = {"Warna daun utuh": D["leaf"], "Warna lesi": D["les"], "Bentuk lesi": D["shape"], "Tekstur": D["tex"],
        "Warna lesi+bentuk": np.hstack([D["les"], D["shape"]]),
        "Warna lesi+bentuk+tekstur": np.hstack([D["les"], D["shape"], D["tex"]])}
split = lambda seed: list(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y))
out = {k: evaluate(v, y, None, split, CL) for k, v in sets.items()}
json.dump(dict(classes=CL, n=len(y), counts=np.bincount(y).tolist(), res=out), open("../results/rice.json", "w"), indent=1)
for k, r in out.items(): print("RICE", k, {m: f"{v['mean']:.3f}±{v['std']:.3f}" for m, v in r.items()})
