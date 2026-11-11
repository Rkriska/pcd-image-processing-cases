import glob, os, json, numpy as np, cv2
from scipy import ndimage as ndi
from sklearn.model_selection import StratifiedKFold
from common import *
CL = ["Unripe", "Ripe", "Overripe"]
def mask_fruit(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = np.concatenate([lab[:6].reshape(-1,3), lab[-6:].reshape(-1,3), lab[:, :6].reshape(-1,3), lab[:, -6:].reshape(-1,3)])
    d = np.linalg.norm(lab - np.median(b, 0), axis=2)
    d8 = np.clip(d / d.max() * 255, 0, 255).astype(np.uint8)
    t, m = cv2.threshold(cv2.GaussianBlur(d8, (7,7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        k = 1 + np.argmax(st[1:, 4]); m = (lb == k)
        m = ndi.binary_fill_holes(m)
    else: m = np.zeros(m.shape, bool)
    if m.mean() < 0.05 or m.mean() > 0.95:
        m = np.zeros(m.shape, bool); h, w = m.shape; m[h//5:4*h//5, w//5:4*w//5] = True
    return m
X_c, X_cu, X_t, y, files, masks = [], [], [], [], [], []
for ci, c in enumerate(CL):
    for f in sorted(glob.glob(f"../data/palm/{c}/*")):
        im = cv2.imread(f); im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
        m = mask_fruit(im); hsv, lab = hsv_lab(im)
        col = np.concatenate([moments(hsv[m][:, [0,1,2,3]]) if False else moments(hsv[m]), moments(lab[m])])
        colu = np.concatenate([moments(hsv.reshape(-1,4)), moments(lab.reshape(-1,3))])
        ys, xs = np.where(m); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[ys.min():ys.max()+1, xs.min():xs.max()+1]
        tex = np.concatenate([glcm_feats(g), lbp_hist(g)])
        X_c.append(col); X_cu.append(colu); X_t.append(tex); y.append(ci); files.append(f); masks.append(m)
y = np.array(y); X_c, X_cu, X_t = map(np.array, (X_c, X_cu, X_t))
np.savez("../results/palm_feats.npz", X_c=X_c, X_cu=X_cu, X_t=X_t, y=y)
np.save("../results/palm_masks.npy", np.array(masks)); json.dump(files, open("../results/palm_files.json","w"))
sets = {"Warna (tanpa mask)": X_cu, "Warna (mask)": X_c, "Tekstur saja": X_t, "Warna+tekstur": np.hstack([X_c, X_t])}
split = lambda seed: list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X_c, y))
out = {k: evaluate(v, y, None, split, CL) for k, v in sets.items()}
json.dump(dict(classes=CL, n=len(y), counts=np.bincount(y).tolist(), res=out), open("../results/palm.json", "w"), indent=1)
for k, r in out.items(): print("PALM", k, {m: f"{v['mean']:.3f}±{v['std']:.3f}" for m, v in r.items()})
