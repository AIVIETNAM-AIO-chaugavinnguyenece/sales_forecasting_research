import gc, numpy as np, pandas as pd, lightgbm as lgbm, xgboost as xgb, shap
from scipy import stats
from sklearn.metrics import mean_absolute_error
import warnings; warnings.filterwarnings("ignore")
SEED=2025; rng=np.random.default_rng(SEED); N_EX=1200; N_BG=80
df=pd.read_parquet("../data/feature_engineered_data_69_features.parquet")
DROP=["date","is_test","store_item","promo_id","store_name","item_name"]
allf=[c for c in df.columns if c not in DROP+["sales"]]
X=df[allf].copy()
for c in allf:
    if str(X[c].dtype)=="category": X[c]=X[c].cat.codes.astype("int32")
X=X.astype("float32")
tm=(~df["is_test"]).values
ytr=df.loc[tm,"sales"].values; yte=df.loc[~tm,"sales"].values
LAGP=("sales_lag_","sales_mean_","sales_min_","sales_max_","sales_std_","sales_ewma_")
lagf=[c for c in allf if c.startswith(LAGP) or c in ["store_mean_7d","item_mean_7d"]]
nolagf=[c for c in allf if c not in lagf]
print(f"with_lags {len(allf)} | no_lags {len(nolagf)} | removed {len(lagf)}",flush=True)
truth=pd.DataFrame([(1,0,-.55),(2,-.004,-.70),(3,0,-.45),(4,-.008,-.60),(5,-.006,-.80),(6,0,-.50),
(7,.002,-.40),(8,0,-.95),(9,.010,-.90),(10,.045,-1.35),(11,-.006,-.75),(12,-.025,-1.00),
(13,.035,-1.45),(14,.020,-1.15),(15,.042,-1.10),(16,-.024,-1.20),(17,-.020,-.85),(18,.050,-1.30),
(19,.015,-1.50),(20,-.005,-1.25),(21,-.018,-1.40),(22,0,-1.15),(23,.004,-1.10),(24,0,-.90),
(25,0,-1.05),(26,0,-.75),(27,0,-.80),(28,.060,-1.10),(29,.045,-.95),(30,-.045,-.70)],
columns=["item_id","te","el"])
truth["ate"]=truth.te.abs(); truth["ael"]=truth.el.abs()
TF=[c for c in ["temp_anomaly","temperature","heat_excess","cold_excess","temp_norm"] if c in allf]
PF=[c for c in ["log_price_ratio","discount_pct","price","base_price","is_deep_discount"] if c in allf]
ei=np.sort(rng.choice(int((~tm).sum()),N_EX,replace=False))
iid=df.loc[~tm].iloc[ei]["item_id"].values
bgi=rng.choice(int(tm.sum()),N_BG,replace=False)
acc={}; mass={}
for arm,cols in [("with_lags",allf),("no_lags",nolagf)]:
    Xtr=X.loc[tm,cols]; Xte=X.loc[~tm,cols]; Xe=Xte.iloc[ei].copy(); bg=Xtr.iloc[bgi].copy()
    for mn,m in [("LightGBM",lgbm.LGBMRegressor(n_estimators=300,num_leaves=63,learning_rate=.05,
        min_child_samples=40,random_state=SEED,n_jobs=2,verbose=-1)),
        ("XGBoost",xgb.XGBRegressor(n_estimators=300,learning_rate=.05,max_depth=8,
        min_child_weight=10,tree_method="hist",enable_categorical=False,random_state=SEED,n_jobs=2,verbosity=0))]:
        m.fit(Xtr,ytr); acc[(arm,mn)]=mean_absolute_error(yte,m.predict(Xte))
        ex=shap.TreeExplainer(m,data=shap.maskers.Independent(bg,max_samples=N_BG),
                              feature_perturbation="interventional")
        sv=np.asarray(ex.shap_values(Xe,check_additivity=False))
        for dv,grp in [("temp",TF),("price",PF)]:
            gi=[cols.index(c) for c in grp if c in cols]
            mass[(arm,mn,dv)]=pd.Series(np.abs(sv[:,gi]).sum(1)).groupby(iid).mean()
        del sv,m,ex; gc.collect()
    del Xtr,Xte,Xe,bg; gc.collect(); print(f"  {arm} done",flush=True)
TC={"temp":"ate","price":"ael"}
print("\n=== ACCURACY (test MAE) ===",flush=True)
for mn in ["LightGBM","XGBoost"]:
    w,n=acc[("with_lags",mn)],acc[("no_lags",mn)]
    print(f"  {mn:9s} with {w:.4f}  no_lags {n:.4f}  cost {100*(n-w)/w:+.1f}%")
print("\n=== DRIVER RECOVERY ===")
for dv in ["temp","price"]:
    for mn in ["LightGBM","XGBoost"]:
        v=[]
        for arm in ["with_lags","no_lags"]:
            d=mass[(arm,mn,dv)].reset_index(); d.columns=["item_id","m"]; d=d.merge(truth,on="item_id")
            r=stats.spearmanr(d.m,d[TC[dv]]); v.append((r.statistic,r.pvalue))
        print(f"  {dv:6s} {mn:9s} with {v[0][0]:+.3f}(p={v[0][1]:.4f})  no_lags {v[1][0]:+.3f}(p={v[1][1]:.4f})  delta {v[1][0]-v[0][0]:+.3f}")
print("\n=== BOOTSTRAP delta rho ===")
br=np.random.default_rng(SEED)
for dv in ["temp","price"]:
    for mn in ["LightGBM","XGBoost"]:
        d=pd.DataFrame({"w":mass[("with_lags",mn,dv)],"n":mass[("no_lags",mn,dv)]}).reset_index(names="item_id").merge(truth,on="item_id").dropna()
        tc=TC[dv]; N=len(d); ds=[]
        for _ in range(1500):
            s=d.iloc[br.integers(0,N,N)]
            if s[tc].nunique()<3: continue
            ds.append(stats.spearmanr(s.n,s[tc]).statistic-stats.spearmanr(s.w,s[tc]).statistic)
        ds=np.array(ds); lo,hi=np.percentile(ds,[2.5,97.5])
        print(f"  {dv:6s} {mn:9s} mean {ds.mean():+.3f}  CI[{lo:+.3f},{hi:+.3f}]  P(>0)={(ds>0).mean():.3f}  sig={bool(lo>0 or hi<0)}")
