import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import metrics
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score,KFold,cross_validate,train_test_split,cross_val_predict


def GetFeaturesImportance(RFR, feature_list):
    # get numerical feature importances
    importances = list(RFR.feature_importances_)

    # list of tuples with variable and importance
    feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]

    # sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    return feature_importances


def OutputImportanceRank(feature_importances, index=5):
    feature_list_sorted = []
    for i, pair in enumerate(feature_importances):
        print('Variable: {:30} Importance: {}'.format(*pair))
        feature_list_sorted.append(pair[0])
        if i >= index:
            print(feature_list_sorted)
            break
        

def PlotImportanceRank(feature_importances):
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.rc('font', family='Times New Roman')
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.weight"] = "bold"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, pair in enumerate(feature_importances):
        if i >= 10:
            break
        rects = ax.bar(*pair, color='lightblue')
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, str(height), 
                     size=24, ha='center', va='bottom', rotation=45, weight='regular')
    # ==========================================================
    ax.set_ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.set_ylabel("Relative Importance", fontsize=28)
    # ax.tick_params(labelsize=18)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(24)         # 设置y轴坐标轴刻度大小
    for tick in ax.get_xticklabels():
        tick.set_rotation(45) 
        tick.set_fontsize(18)         # 设置x轴坐标轴刻度大小
    plt.tight_layout()
    plt.savefig("./rank.png", bbox_inches='tight', dpi=480)
    plt.show()


def PlotCrossValPredict(RFR, X_total, y_total):
    predicted = cross_val_predict(RFR, X_total, y_total, cv=kfold)

    fig, ax = plt.subplots()
    ax.scatter(y_total, predicted, edgecolors=(0, 0, 0))
    ax.plot([y_total.min(), y_total.max()], [y_total.min(), y_total.max()], "k--", lw=4)
    ax.set_xlabel("Measured U (eV)")
    ax.set_ylabel("Predicted U (eV)")
    plt.show()


def AdjustEstimator(X_total, y_total):  
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, shuffle=True)

    estimators_list = list(range(8,100))
    mae_list = []
    rmse_list = []
    r2_list = []
    for estimator in estimators_list:
        RFR=RandomForestRegressor(random_state=0,n_jobs=8,n_estimators=estimator)
        RFR.fit(X_train, y_train)
        predictions = RFR.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
        r2 = metrics.r2_score(y_test, predictions)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
    plt.plot(estimators_list, mae_list, label="MAE (eV)")
    plt.plot(estimators_list, r2_list, label="$R^{2}$")
    plt.plot(estimators_list, rmse_list, label="RMSE (eV)")
    plt.legend()
    plt.show()


def Plot(RFR, X_total, y_total):         
    RFR.fit(X_total, y_total)
    #========================= plot the results =====================================
    # predictions = RFR.predict(X_total)
    predictions = RFR.oob_prediction_

    # plot the predictions against true values
    fig, ax = plt.subplots()

    # ax.hist2d(y_total, predictions, norm=LogNorm(), bins=64, cmap='Blues', alpha=0.9)
    ax.plot(y_total, predictions, "bo")

    ax.set_xlim(ax.get_ylim())
    ax.set_ylim(ax.get_xlim())

    mae = metrics.mean_absolute_error(y_total, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_total, predictions))
    r2 = metrics.r2_score(y_total, predictions)
    ax.text(0.75, 0.1, 'MAE: {:.2f} eV \nRMSE: {:.2f} eV \n$R^2$: {:.2f}'.format(mae, rmse, r2),
            transform=ax.transAxes,
        bbox={'facecolor': 'w', 'edgecolor': 'k'})

    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    ax.fill_between(ax.get_xlim(), 
                    np.array(ax.get_ylim())-0.5, 
                    np.array(ax.get_ylim())+0.5,
                    facecolor='c', 
                    alpha=0.6,
                    label="error < 0.5 eV")

    ax.set_xlabel('DFT $U$ (eV)')
    ax.set_ylabel('ML $U$ (eV)')

    fig.tight_layout()
    plt.legend()
    plt.show()


# =====================read the raw data===============
data=pd.read_csv('./MLDatasets.csv')         # Read the data which contains descriptors and targets
X_total=data.iloc[:, 6:-1]                   # Select the feature columns
feature_list = list(X_total.columns)
X_total = data[feature_list]
S=StandardScaler()
X_total=S.fit_transform(X_total)
y_total=data['target']
# ==============set the picture type=================
plt.rcParams['figure.figsize'] = (6.0, 6.0)
plt.rc('font', family='Times New Roman')
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
# ==============take all the data as the training set=======
RFR=RandomForestRegressor(random_state=0,n_jobs=8,n_estimators=64, oob_score=True)
Plot(RFR, X_total, y_total)
feature_importances = GetFeaturesImportance(RFR, feature_list)
PlotImportanceRank(feature_importances)
OutputImportanceRank(feature_importances, index=9)
quit()
#===================== save predictions ===================
# pred = pd.DataFrame(data=[data["filepath"],
#                     data["structures"],
#                     y_total,
#                     RFR.oob_prediction_,
#                     data["observation"]],
#                     index=["filepath", "structures", "target", "pred", "observation"]).T
# pred.to_csv("./id_pred.csv")
# quit()
# #=============================KFold training and test====================================
n_splits = 5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=242)   # random selection
model_list=[RFR]
model_name=['RFR']
score_list=['r2','neg_mean_absolute_error','neg_mean_squared_error']
for i in range(len(model_list)):
    s_list=[]
    ten_fold=pd.DataFrame()
    ten_fold['n']=range(1, n_splits+1)
    for score in score_list:
        #s=cross_val_score(model_list[i],X_total,y_total,scoring=score,cv=kfold,n_jobs=8)
        s=cross_validate(model_list[i],X_total,y_total,scoring=score,cv=kfold,n_jobs=8,return_train_score=True,
                        return_estimator=True)

        if score=='r2':
            ten_fold['R2_test']=s['test_score']
            ten_fold['R2_train']=s['train_score']
        elif score=='neg_mean_absolute_error':
            ten_fold['MAE_test']=[-i for i in s['test_score']]
            ten_fold['MAE_train']=[-i for i in s['train_score']]
        else:
            ten_fold['MSE_test']=[-i for i in s['test_score']]
            ten_fold['MSE_train']=[-i for i in s['train_score']]
    ten_fold.to_csv(model_name[i]+'withNormalize_5fold.csv')
    #===========================================================================
    for estimator in s['estimator']:
        feature_importances = GetFeaturesImportance(estimator, feature_list)
        PlotImportanceRank(feature_importances)
        OutputImportanceRank(feature_importances)
        # print(estimator.oob_score_)
        print("=====================================================")
#==================================================================================
# PlotCrossValPredict(RFR, X_total, y_total)


