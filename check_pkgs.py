import sys
pkgs=["sklearn","imblearn","shap","xgboost","lightgbm"]
for p in pkgs:
    try:
        m=__import__(p)
        print("OK",p)
    except: print("MISSING",p)

