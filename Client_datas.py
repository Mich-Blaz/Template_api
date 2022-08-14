# -*- coding: utf-8 -*-
"""
Récupération des données sous la forme suivante pour l'inclure dans l'API
@author: MichBlaz
"""
from pydantic import BaseModel 
class Client_data(BaseModel):
   NAME_CONTRACT_TYPE: str
   CODE_GENDER: str
   FLAG_OWN_CAR: str
   FLAG_OWN_REALTY: str
   CNT_CHILDREN: int
   AMT_INCOME_TOTAL: float
   AMT_CREDIT: float
   AMT_ANNUITY: float
   NAME_TYPE_SUITE: str
   NAME_INCOME_TYPE: str
   NAME_EDUCATION_TYPE: str
   NAME_FAMILY_STATUS: str
   DAYS_BIRTH: int
   DAYS_ID_PUBLISH: int
   OWN_CAR_AGE: float
   FLAG_MOBIL: int
   FLAG_WORK_PHONE: int
   FLAG_CONT_MOBILE: int
   FLAG_PHONE: int
   FLAG_EMAIL: int
   OCCUPATION_TYPE: str
   CNT_FAM_MEMBERS: float
   WEEKDAY_APPR_PROCESS_START: str
   REG_REGION_NOT_LIVE_REGION: int
   LIVE_REGION_NOT_WORK_REGION: int
   REG_CITY_NOT_LIVE_CITY: int
   REG_CITY_NOT_WORK_CITY: int
   ORGANIZATION_TYPE: str
   EXT_SOURCE_2: float
   BASEMENTAREA_AVG: float
   YEARS_BEGINEXPLUATATION_AVG: float
   YEARS_BUILD_AVG: float
   COMMONAREA_AVG: float
   ELEVATORS_AVG: float
   ENTRANCES_AVG: float
   FLOORSMAX_AVG: float
   FLOORSMIN_AVG: float
   LANDAREA_AVG: float
   LIVINGAPARTMENTS_AVG: float
   LIVINGAREA_AVG: float
   NONLIVINGAPARTMENTS_AVG: float
   NONLIVINGAREA_AVG: float
   FONDKAPREMONT_MODE: str
   HOUSETYPE_MODE: str
   TOTALAREA_MODE: float
   WALLSMATERIAL_MODE: str
   EMERGENCYSTATE_MODE: str
   OBS_30_CNT_SOCIAL_CIRCLE: float
   DEF_30_CNT_SOCIAL_CIRCLE: float
   DEF_60_CNT_SOCIAL_CIRCLE: float
   DAYS_LAST_PHONE_CHANGE: float
   FLAG_DOCUMENT_3: int
   FLAG_DOCUMENT_8: int
   FLAG_DOCUMENT_11: int
   FLAG_DOCUMENT_13: int
   FLAG_DOCUMENT_16: int
   FLAG_DOCUMENT_18: int
   AMT_REQ_CREDIT_BUREAU_MON: float
   AMT_REQ_CREDIT_BUREAU_QRT: float
   AMT_REQ_CREDIT_BUREAU_YEAR: float
   INCOME_CREDIT_PERC: float
   INCOME_PER_PERSON: float
   ANNUITY_INCOME_PERC: float
   PAYMENT_RATE: float
   BURO_DAYS_CREDIT_MIN: float
   BURO_DAYS_CREDIT_MAX: float
   BURO_DAYS_CREDIT_MEAN: float
   BURO_DAYS_CREDIT_VAR: float
   BURO_DAYS_CREDIT_ENDDATE_MIN: float
   BURO_DAYS_CREDIT_ENDDATE_MAX: float
   BURO_DAYS_CREDIT_ENDDATE_MEAN: float
   BURO_DAYS_CREDIT_UPDATE_MEAN: float
   BURO_CREDIT_DAY_OVERDUE_MAX: float
   BURO_AMT_CREDIT_MAX_OVERDUE_MEAN: float
   BURO_AMT_CREDIT_SUM_MAX: float
   BURO_AMT_CREDIT_SUM_MEAN: float
   BURO_AMT_CREDIT_SUM_SUM: float
   BURO_AMT_CREDIT_SUM_DEBT_MAX: float
   BURO_AMT_CREDIT_SUM_DEBT_MEAN: float
   BURO_AMT_CREDIT_SUM_OVERDUE_MEAN: float
   BURO_AMT_CREDIT_SUM_LIMIT_MEAN: float
   BURO_AMT_CREDIT_SUM_LIMIT_SUM: float
   BURO_AMT_ANNUITY_MAX: float
   BURO_MONTHS_BALANCE_SIZE_MEAN: float
   BURO_MONTHS_BALANCE_SIZE_SUM: float
   BURO_CREDIT_ACTIVE_Active_MEAN: float
   BURO_CREDIT_ACTIVE_Sold_MEAN: float
   BURO_CREDIT_TYPE_Another_type_of_loan_MEAN: float
   BURO_CREDIT_TYPE_Car_loan_MEAN: float
   BURO_CREDIT_TYPE_Consumer_credit_MEAN: float
   BURO_CREDIT_TYPE_Credit_card_MEAN: float
   BURO_CREDIT_TYPE_Microloan_MEAN: float
   BURO_CREDIT_TYPE_Mortgage_MEAN: float
   BURO_STATUS_0_MEAN_MEAN: float
   BURO_STATUS_1_MEAN_MEAN: float
   BURO_STATUS_C_MEAN_MEAN: float
   BURO_STATUS_X_MEAN_MEAN: float
   ACTIVE_DAYS_CREDIT_MIN: float
   ACTIVE_DAYS_CREDIT_MAX: float
   ACTIVE_DAYS_CREDIT_MEAN: float
   ACTIVE_DAYS_CREDIT_VAR: float
   ACTIVE_DAYS_CREDIT_ENDDATE_MIN: float
   ACTIVE_DAYS_CREDIT_ENDDATE_MAX: float
   ACTIVE_DAYS_CREDIT_ENDDATE_MEAN: float
   ACTIVE_DAYS_CREDIT_UPDATE_MEAN: float
   ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN: float
   ACTIVE_AMT_CREDIT_SUM_MAX: float
   ACTIVE_AMT_CREDIT_SUM_MEAN: float
   ACTIVE_AMT_CREDIT_SUM_SUM: float
   ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN: float
   ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN: float
   ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN: float
   ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM: float
   ACTIVE_AMT_ANNUITY_MAX: float
   ACTIVE_MONTHS_BALANCE_MAX_MAX: float
   ACTIVE_MONTHS_BALANCE_SIZE_MEAN: float
   CLOSED_DAYS_CREDIT_MAX: float
   CLOSED_DAYS_CREDIT_MEAN: float
   CLOSED_DAYS_CREDIT_VAR: float
   CLOSED_DAYS_CREDIT_ENDDATE_MIN: float
   CLOSED_DAYS_CREDIT_ENDDATE_MAX: float
   CLOSED_DAYS_CREDIT_ENDDATE_MEAN: float
   CLOSED_DAYS_CREDIT_UPDATE_MEAN: float
   CLOSED_AMT_CREDIT_SUM_MAX: float
   CLOSED_AMT_CREDIT_SUM_MEAN: float
   CLOSED_AMT_CREDIT_SUM_SUM: float
   CLOSED_AMT_CREDIT_SUM_DEBT_MAX: float
   CLOSED_AMT_CREDIT_SUM_DEBT_MEAN: float
   CLOSED_AMT_CREDIT_SUM_LIMIT_SUM: float
   CLOSED_AMT_ANNUITY_MAX: float
   CLOSED_AMT_ANNUITY_MEAN: float
   CLOSED_MONTHS_BALANCE_SIZE_MEAN: float
   PREV_AMT_ANNUITY_MIN: float
   PREV_AMT_ANNUITY_MAX: float
   PREV_AMT_ANNUITY_MEAN: float
   PREV_AMT_APPLICATION_MIN: float
   PREV_AMT_APPLICATION_MAX: float
   PREV_AMT_APPLICATION_MEAN: float
   PREV_APP_CREDIT_PERC_MIN: float
   PREV_APP_CREDIT_PERC_MAX: float
   PREV_APP_CREDIT_PERC_MEAN: float
   PREV_APP_CREDIT_PERC_VAR: float
   PREV_AMT_DOWN_PAYMENT_MIN: float
   PREV_AMT_DOWN_PAYMENT_MAX: float
   PREV_AMT_DOWN_PAYMENT_MEAN: float
   PREV_AMT_GOODS_PRICE_MIN: float
   PREV_AMT_GOODS_PRICE_MEAN: float
   PREV_HOUR_APPR_PROCESS_START_MIN: float
   PREV_HOUR_APPR_PROCESS_START_MAX: float
   PREV_HOUR_APPR_PROCESS_START_MEAN: float
   PREV_RATE_DOWN_PAYMENT_MIN: float
   PREV_RATE_DOWN_PAYMENT_MAX: float
   PREV_RATE_DOWN_PAYMENT_MEAN: float
   PREV_DAYS_DECISION_MIN: float
   PREV_DAYS_DECISION_MAX: float
   PREV_DAYS_DECISION_MEAN: float
   PREV_CNT_PAYMENT_MEAN: float
   PREV_CNT_PAYMENT_SUM: float
   PREV_NAME_CONTRACT_TYPE_Cash_loans_MEAN: float
   PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN: float
   PREV_NAME_CONTRACT_TYPE_Revolving_loans_MEAN: float
   PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN: float
   PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN: float
   PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN: float
   PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN: float
   PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN: float
   PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN: float
   PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN: float
   PREV_NAME_CASH_LOAN_PURPOSE_Buying_a_home_MEAN: float
   PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN: float
   PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN: float
   PREV_NAME_CONTRACT_STATUS_Approved_MEAN: float
   PREV_NAME_CONTRACT_STATUS_Canceled_MEAN: float
   PREV_NAME_CONTRACT_STATUS_Refused_MEAN: float
   PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN: float
   PREV_CODE_REJECT_REASON_HC_MEAN: float
   PREV_CODE_REJECT_REASON_SCO_MEAN: float
   PREV_CODE_REJECT_REASON_SCOFR_MEAN: float
   PREV_CODE_REJECT_REASON_VERIF_MEAN: float
   PREV_CODE_REJECT_REASON_XAP_MEAN: float
   PREV_NAME_TYPE_SUITE_Family_MEAN: float
   PREV_NAME_TYPE_SUITE_Other_A_MEAN: float
   PREV_NAME_TYPE_SUITE_Other_B_MEAN: float
   PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN: float
   PREV_NAME_TYPE_SUITE_nan_MEAN: float
   PREV_NAME_CLIENT_TYPE_New_MEAN: float
   PREV_NAME_CLIENT_TYPE_Refreshed_MEAN: float
   PREV_NAME_CLIENT_TYPE_Repeater_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Clothing_and_Accessories_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Consumer_Electronics_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Direct_Sales_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Furniture_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Homewares_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN: float
   PREV_NAME_GOODS_CATEGORY_Tourism_MEAN: float
   PREV_NAME_PORTFOLIO_Cards_MEAN: float
   PREV_NAME_PORTFOLIO_Cash_MEAN: float
   PREV_NAME_PRODUCT_TYPE_XNA_MEAN: float
   PREV_NAME_PRODUCT_TYPE_walk_in_MEAN: float
   PREV_NAME_PRODUCT_TYPE_x_sell_MEAN: float
   PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN: float
   PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN: float
   PREV_CHANNEL_TYPE_Contact_center_MEAN: float
   PREV_CHANNEL_TYPE_Country_wide_MEAN: float
   PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN: float
   PREV_CHANNEL_TYPE_Regional___Local_MEAN: float
   PREV_CHANNEL_TYPE_Stone_MEAN: float
   PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN: float
   PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN: float
   PREV_NAME_SELLER_INDUSTRY_Construction_MEAN: float
   PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN: float
   PREV_NAME_SELLER_INDUSTRY_XNA_MEAN: float
   PREV_NAME_YIELD_GROUP_XNA_MEAN: float
   PREV_NAME_YIELD_GROUP_high_MEAN: float
   PREV_NAME_YIELD_GROUP_low_action_MEAN: float
   PREV_NAME_YIELD_GROUP_low_normal_MEAN: float
   PREV_NAME_YIELD_GROUP_middle_MEAN: float
   PREV_PRODUCT_COMBINATION_Card_Street_MEAN: float
   PREV_PRODUCT_COMBINATION_Cash_MEAN: float
   PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN: float
   PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN: float
   PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN: float
   PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN: float
   PREV_PRODUCT_COMBINATION_Cash_X_Sell__middle_MEAN: float
   PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN: float
   PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN: float
   PREV_PRODUCT_COMBINATION_POS_industry_without_interest_MEAN: float
   APPROVED_AMT_ANNUITY_MIN: float
   APPROVED_AMT_ANNUITY_MAX: float
   APPROVED_AMT_ANNUITY_MEAN: float
   APPROVED_AMT_APPLICATION_MIN: float
   APPROVED_AMT_APPLICATION_MAX: float
   APPROVED_AMT_APPLICATION_MEAN: float
   APPROVED_APP_CREDIT_PERC_MIN: float
   APPROVED_APP_CREDIT_PERC_MAX: float
   APPROVED_APP_CREDIT_PERC_MEAN: float
   APPROVED_APP_CREDIT_PERC_VAR: float
   APPROVED_AMT_DOWN_PAYMENT_MIN: float
   APPROVED_AMT_DOWN_PAYMENT_MAX: float
   APPROVED_HOUR_APPR_PROCESS_START_MAX: float
   APPROVED_RATE_DOWN_PAYMENT_MIN: float
   APPROVED_DAYS_DECISION_MAX: float
   APPROVED_DAYS_DECISION_MEAN: float
   APPROVED_CNT_PAYMENT_MEAN: float
   APPROVED_CNT_PAYMENT_SUM: float
   REFUSED_AMT_ANNUITY_MEAN: float
   REFUSED_AMT_APPLICATION_MAX: float
   REFUSED_AMT_APPLICATION_MEAN: float
   REFUSED_APP_CREDIT_PERC_VAR: float
   REFUSED_AMT_DOWN_PAYMENT_MAX: float
   REFUSED_HOUR_APPR_PROCESS_START_MIN: float
   REFUSED_HOUR_APPR_PROCESS_START_MEAN: float
   REFUSED_RATE_DOWN_PAYMENT_MEAN: float
   REFUSED_DAYS_DECISION_MIN: float
   REFUSED_DAYS_DECISION_MAX: float
   REFUSED_DAYS_DECISION_MEAN: float
   REFUSED_CNT_PAYMENT_SUM: float
   POS_MONTHS_BALANCE_MAX: float
   POS_MONTHS_BALANCE_MEAN: float
   POS_MONTHS_BALANCE_SIZE: float
   POS_SK_DPD_MAX: float
   POS_SK_DPD_MEAN: float
   POS_SK_DPD_DEF_MAX: float
   POS_NAME_CONTRACT_STATUS_Active_MEAN: float
   POS_NAME_CONTRACT_STATUS_Completed_MEAN: float
   POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN: float
   POS_NAME_CONTRACT_STATUS_Signed_MEAN: float
   INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE: float
   INSTAL_DPD_MAX: float
   INSTAL_DPD_MEAN: float
   INSTAL_DPD_SUM: float
   INSTAL_DBD_MAX: float
   INSTAL_DBD_MEAN: float
   INSTAL_DBD_SUM: float
   INSTAL_PAYMENT_PERC_MAX: float
   INSTAL_PAYMENT_PERC_MEAN: float
   INSTAL_PAYMENT_PERC_VAR: float
   INSTAL_PAYMENT_DIFF_MAX: float
   INSTAL_PAYMENT_DIFF_MEAN: float
   INSTAL_PAYMENT_DIFF_SUM: float
   INSTAL_PAYMENT_DIFF_VAR: float
   INSTAL_AMT_INSTALMENT_MAX: float
   INSTAL_AMT_INSTALMENT_MEAN: float
   INSTAL_AMT_INSTALMENT_SUM: float
   INSTAL_AMT_PAYMENT_MIN: float
   INSTAL_DAYS_ENTRY_PAYMENT_MAX: float
   INSTAL_DAYS_ENTRY_PAYMENT_MEAN: float
   INSTAL_DAYS_ENTRY_PAYMENT_SUM: float
   INSTAL_COUNT: float
   CC_MONTHS_BALANCE_MIN: float
   CC_AMT_BALANCE_MIN: float
   CC_AMT_BALANCE_MAX: float
   CC_AMT_BALANCE_MEAN: float
   CC_AMT_BALANCE_VAR: float
   CC_AMT_CREDIT_LIMIT_ACTUAL_MIN: float
   CC_AMT_CREDIT_LIMIT_ACTUAL_MAX: float
   CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN: float
   CC_AMT_CREDIT_LIMIT_ACTUAL_SUM: float
   CC_AMT_CREDIT_LIMIT_ACTUAL_VAR: float
   CC_AMT_DRAWINGS_ATM_CURRENT_MAX: float
   CC_AMT_DRAWINGS_ATM_CURRENT_MEAN: float
   CC_AMT_DRAWINGS_ATM_CURRENT_SUM: float
   CC_AMT_DRAWINGS_ATM_CURRENT_VAR: float
   CC_AMT_DRAWINGS_CURRENT_MEAN: float
   CC_AMT_DRAWINGS_CURRENT_VAR: float
   CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN: float
   CC_AMT_DRAWINGS_POS_CURRENT_MEAN: float
   CC_AMT_DRAWINGS_POS_CURRENT_SUM: float
   CC_AMT_DRAWINGS_POS_CURRENT_VAR: float
   CC_AMT_INST_MIN_REGULARITY_VAR: float
   CC_AMT_PAYMENT_CURRENT_MEAN: float
   CC_AMT_PAYMENT_CURRENT_SUM: float
   CC_AMT_PAYMENT_CURRENT_VAR: float
   CC_CNT_DRAWINGS_ATM_CURRENT_MEAN: float
   CC_CNT_DRAWINGS_ATM_CURRENT_SUM: float
   CC_CNT_DRAWINGS_ATM_CURRENT_VAR: float
   CC_CNT_DRAWINGS_CURRENT_MAX: float
   CC_CNT_DRAWINGS_CURRENT_MEAN: float
   CC_CNT_DRAWINGS_CURRENT_VAR: float
   CC_CNT_DRAWINGS_POS_CURRENT_MIN: float
   CC_CNT_INSTALMENT_MATURE_CUM_MAX: float
   CC_CNT_INSTALMENT_MATURE_CUM_VAR: float
   CC_NAME_CONTRACT_STATUS_Completed_VAR: float