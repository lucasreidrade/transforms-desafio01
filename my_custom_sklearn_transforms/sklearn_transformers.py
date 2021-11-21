from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from math import log

# Um transformador para remover colunas indesejadas
class CustomTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do DataFrame 'X' de entrada
        data = X.copy()
        
        features = [
            "CHECKING_BALANCE",
            "CREDIT_HISTORY",
            "CURRENT_RESIDENCE_DURATION",
            "EMPLOYMENT_DURATION",
            "EXISTING_CREDITS_COUNT",
            "EXISTING_SAVINGS",
            "HOUSING",
            "LOAN_PURPOSE",
            "LOAN_AMOUNT",
            "INSTALLMENT_PERCENT",
            "INSTALLMENT_PLANS",
            "PAYMENT_TERM",
            "OTHERS_ON_LOAN",
            "PROPERTY",
            "SEX",
            "TELEPHONE"
        ]
        values=[0.0,"UNKNOWN",3.0,4.0,0.0,33.56,"OWN","CAR_NEW", 3237.0, 3.0,"NONE",651.0,"NONE","SAVINGS_INSURANCE","M",0.0]
        
        for f in features:
            if f not in data.columns:
                data[f]= float("Nan")
                
        data["CHECKING_BALANCE_STATUS_UNKNOWN"]   = data["CHECKING_BALANCE"].isna()
        data["EXISTING_SAVINGS_STATUS_UNKNOWN"]   = data["EXISTING_SAVINGS"].isna() 
        
        for f,v in zip(features,values):
            data[[f]].fillna(v)
            
        data["CREDIT_HISTORY_OUTSTANDING_CREDIT"] = data["CREDIT_HISTORY"]    == "OUTSTANDING_CREDIT"
        data["HOUSING_OWN"]                       = data["HOUSING"]           == "OWN"
        data["INSTALLMENT_PLANS_BANK"]            = data["INSTALLMENT_PLANS"] == "BANK"           
        data["INSTALLMENT_PLANS_NONE"]            = data["INSTALLMENT_PLANS"] == "NONE"             
        data["LOAN_PURPOSE_BUSINESS"]             = data["LOAN_PURPOSE"]      == "BUSINESS"       
        data["LOAN_PURPOSE_RETRAINING"]           = data["LOAN_PURPOSE"]      == "RETRAINING"         
        data["LOAN_PURPOSE_VACATION"]             = data["LOAN_PURPOSE"]      == "VACATION"            
        data["OTHERS_ON_LOAN_NONE"]               = data["OTHERS_ON_LOAN"]    == "NONE"              
        data["PROPERTY_CAR_OTHER"]                = data["PROPERTY"]          == "CAR_OTHER"               
        data["PROPERTY_REAL_ESTATE"]              = data["PROPERTY"]          == "REAL_ESTATE"
        data["PROPERTY_UNKNOWN"]                  = data["PROPERTY"]          == "UNKNOWN"
        data["SEX_M"]                             = data["SEX"]               == "M"
        
        data["LOG_LOAN_INTERESTS"] = (1+data["INSTALLMENT_PERCENT"]/100.0)
        data["LOG_LOAN_INTERESTS"] = data["LOG_LOAN_INTERESTS"]**(data["PAYMENT_TERM"]/30)
        data["LOG_LOAN_INTERESTS"] = data["LOG_LOAN_INTERESTS"]-1
        data["LOG_LOAN_INTERESTS"] = data["LOG_LOAN_INTERESTS"]*data['LOAN_AMOUNT']
        data["LOG_LOAN_INTERESTS"] = data["LOG_LOAN_INTERESTS"].apply(lambda x: log(x))
        
        filtered = [
            "CHECKING_BALANCE",
            "CHECKING_BALANCE_STATUS_UNKNOWN",
            "CREDIT_HISTORY_OUTSTANDING_CREDIT",
            "CURRENT_RESIDENCE_DURATION",
            "EMPLOYMENT_DURATION",
            "EXISTING_CREDITS_COUNT",
            "EXISTING_SAVINGS",
            "EXISTING_SAVINGS_STATUS_UNKNOWN",
            "HOUSING_OWN",
            "INSTALLMENT_PERCENT",
            "INSTALLMENT_PLANS_BANK",
            "INSTALLMENT_PLANS_NONE",
            "LOAN_PURPOSE_BUSINESS",
            "LOAN_PURPOSE_RETRAINING",
            "LOAN_PURPOSE_VACATION",
            "LOG_LOAN_INTERESTS",
            "OTHERS_ON_LOAN_NONE",
            "PROPERTY_CAR_OTHER",
            "PROPERTY_REAL_ESTATE",
            "PROPERTY_UNKNOWN",
            "SEX_M",
            "TELEPHONE"
            ]
            
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data[filtered]
