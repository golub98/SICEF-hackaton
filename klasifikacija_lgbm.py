import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import lightgbm as lgb

path = r"C:\Users\Darko\Desktop\Darko\Master etf\Masinsko ucenje\lightGBM\data\corona_tested_individuals_ver_006.english.csv"


def izbaci_suvisne_kolone(df):
    """
    Izbacujem neinformativne atribute. Ostaje da se ispita kad se bude imalo vremena da li su ovi atributi
    (pol,godine iznad 60) mozda informativni iako na prvi pogled ne deluje da  su korelisani sa korona virusom
    :param df: dataframe
    :return: izmenjen dataframe
    """
    return df.drop(labels=["test_date", "age_60_and_above", "gender",], axis=1)



def dropuj_cilj_vr(df, kolona, vrednost):
    """
    iz  data frame-a za zadatu kolonu kolona brise redove koji imaju vrednost te kolone - vrednost
    :param df: dataframe
    :param kolona: string ime kolone
    :param vrednost: vrednost koju trazimo
    :return: izmenjen dataframe
    """
    return df.drop(df[df[kolona] == vrednost].index)


def dropuj_nedostajuce(df):
    """
    Izbacuje nedostajuce vrednosti iz df-a. Ovde je specificno sto je u kolonama None zapravo string pa
    smo ga morali ovako rucno dropovati. inace bismo samo koristili df.dropna
    :param df: dataframe
    :return: izmenjeni dataframe
    """
    df = dropuj_cilj_vr(df, kolona="cough", vrednost="None")
    # print(df["cough"].value_counts())
    df = dropuj_cilj_vr(df, kolona="fever", vrednost="None")
    # print(df["fever"].value_counts())

    df = dropuj_cilj_vr(df, kolona="sore_throat", vrednost="None")
    # print(df["sore_throat"].value_counts())

    df = dropuj_cilj_vr(df, kolona="shortness_of_breath", vrednost="None")
    # print(df["shortness_of_breath"].value_counts())

    df = dropuj_cilj_vr(df, kolona="head_ache", vrednost="None")
    # print(df["head_ache"].value_counts())

    df = dropuj_cilj_vr(df, kolona="contact", vrednost="Abroad")
    # print(df["contact"].value_counts())

    df = dropuj_cilj_vr(df, kolona="label", vrednost="other")
    return df


def kodiranje_izlaza(y):
    """
    Kodira izlaz negativan na koronu -> 0, pozitivan->1
    :param y: niz izlaza
    :return: niz kodiranih izlaza
    """
    # Kodiranje kategorickog izlaza
    labelencoder = LabelEncoder()
    Y = labelencoder.fit_transform(y)  # negativan na koronu -> 0, pozitivan->1
    return Y


def kodiranje_ulaza(df):
    """
    Kodira jedinu kategoricku promenljivu u numericku (contact)
    :param df: Daaframe
    :return: izmenjeni Dataframe
    """
    ord_enc = OrdinalEncoder()
    #print("pre binarizovanja")
    #print(df.contact.head(10))

    df["contact"] = ord_enc.fit_transform(df[["contact"]]) # ovaj da da je 0-kontakt 1-nije kontakt a ja
    # hocu suprotno
    df["contact"] = (df["contact"]+1)%2
    #print("posle binarizovanja")
    #print(df.contact.head(10))
    X = np.array(df.values)
    return X.astype(np.float)


def predprocesiranje_podataka(path):
    """
    ova funkcija se uvek razlikuje od baze do baze jer su baze razlicito popunjavane.
    Cilj je izbaciti redove koji imaju nedostajuce vrednosti, izbaciti nepozeljne kolone, prebaciti
    promenljive iz kategorickih u numericke.
    :param path: adresa na kojoj se nalazi csv fajl sa tabelom
    :return: X_train, X_test, y_train, y_test
    """
    # importujemo dataset
    df = pd.read_csv(path)
    # print(df.columns)
    df = df.sample(frac=1).reset_index(drop=True)  # random promesa dataset dok je sve u jednom

    # Preimenuj corona_result u label
    df = df.rename(columns={'corona_result': 'label'})
    # Preimenuj test_indication u contact
    df = df.rename(columns={'test_indication': 'contact'})
    df = izbaci_suvisne_kolone(df)  # ne uzimamo u obzir test_date, age_60_and_above i gender za klasifikaciju

    df = dropuj_nedostajuce(df)

    x = df.drop(labels=["label"], axis=1)  # atributi za klasifikaciju, josuvek kategoricki
    X = kodiranje_ulaza(x)  # prebacuje kategoricke promenljive u numericke i vraca X kao np.array

    y = df["label"].values
    Y = kodiranje_izlaza(y)  # ovo je sad np.array 0 ili 1 (1 pozitivan na koronu)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def lgb_f1_score(y_hat,data):
    """
    racuna f1 skor i vraca ga u formatu koji odgovara funkciji train. kako je ovo custom metrika mora da se
    postuju ulazni parametri koje zahteva i njihovi tipovi (obrati paznju na data)
    :param y_hat: estimirani izlaz
    :param data: specificiran zeljeni izlaz
    :return: score u zahtevanom formatu
    """
    y_true = data.get_label()
    y_hat = np.round(y_hat)
    return 'f1', f1_score(y_true, y_hat,average='macro'), True

def klasifikator(x):
    """
    :param x: ovo je ulaz u obliku [cough,fever,sore_throat,shortness_of_breath,head_ache, contact]
    :return: verovatnoca da ima koronu
    """
    x = np.fromstring(x, dtype=int, sep=' ')
    x = [x]
    model = lgb.Booster(model_file='lgbm_model_najbolji.txt')
    return model.predict(x)

def vizualizacija_klasifikacije(y_test, y_predikt):
    pozitivni = y_predikt[y_test==1]
    negativni = y_predikt[y_test==0]
    bins = np.linspace(0, 1, 20)

    plt.hist(pozitivni, bins, alpha=0.5, label='poz')
    plt.title('histogram pozitivnih na COVID19')
    plt.xlabel("verovatnoca na izlazu klasifikatora")
    plt.show()

    plt.hist(negativni, bins, alpha=0.5, label='neg')
    plt.title('histogram negativnih na COVID19')
    plt.xlabel("verovatnoca na izlazu klasifikatora")
    plt.show()

def formiraj_lgbm_model():
    X_train, X_test, y_train, y_test = predprocesiranje_podataka(path)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    d_train = lgb.Dataset(X_train, label=y_train)
    #d_val = lgb.Dataset(X_val, label=y_val)
    # lako dodam early stopping koristeci validacioni skup

    lgbm_params = {'learning_rate': 0.05, 'boosting_type': 'gbdt',  # probaj nekad "dart" za polju preciznost. nekad radi
                   'objective': 'binary',
                   'metric': ['auc','binary_logloss'],
                   'num_leaves': 20,
                   'max_depth': 10,
                   'num_iterations': 100,

                   }

    # ako koristim custom evaluacionu funkciju feval=lgb_f1_score
    clf = lgb.train(params=lgbm_params, train_set=d_train, feval=lgb_f1_score, num_boost_round=50)
    #clf.save_model("lgbm_model_najbolji.txt",num_iteration = clf.best_iteration) # save modell
    #model = lgb.Booster(model_file='lgbm_model.txt') # ovako se model ucitava

    # Predikcija na testirajucem skupu
    y_pred_lgbm = clf.predict(X_test)

    vizualizacija_klasifikacije(y_test, y_pred_lgbm)
    # Hard limituj ga na 0/1 ako bas mora da se kaze binarno zdrav/bolestan
    for i in range(0, X_test.shape[0]):
        if y_pred_lgbm[i] >= .35: # pomerena granica ka zarazenim jer mi je veca cena za missdetection
            # u prevodu gore je da bolesnog coveka klasifikujes kao zdravog nego obrnuto
            y_pred_lgbm[i] = 1
        else:
            y_pred_lgbm[i] = 0


    # Matrica konfuzije
    cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
    sns.heatmap(cm_lgbm, annot=True)

    # F1 score je dobar pokazatelj performansi jer ne diskriminise manje zastupljene klase
    f1_skor = f1_score(y_test, y_pred_lgbm, average='macro')

    print("Accuracy with LGBM = ", metrics.accuracy_score(y_pred_lgbm, y_test)) # preciznost, ali nije najinformativnija
    print("AUC score with LGBM is: ", roc_auc_score(y_pred_lgbm, y_test)) # auc skor

if __name__ == "__main__":
    formiraj_lgbm_model()
    x = '1 1 0 0 0 1'
    print(klasifikator(x))






