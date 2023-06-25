CREATE TABLE pos_cash_balance (
  SK_ID_PREV integer NOT NULL,
  SK_ID_CURR integer NOT NULL,
  MONTHS_BALANCE smallint NOT NULL,
  CNT_INSTALMENT real NULL,
  CNT_INSTALMENT_FUTURE real NULL,
  NAME_CONTRACT_STATUS varchar(21) NOT NULL,
  SK_DPD smallint NOT NULL,
  SK_DPD_DEF smallint NOT NULL);

CREATE TABLE application_united (
  SK_ID_CURR integer NOT NULL,
  TARGET real NULL,
  NAME_CONTRACT_TYPE varchar(15) NOT NULL,
  CODE_GENDER varchar(3) NOT NULL,
  FLAG_OWN_CAR char(1) NOT NULL,
  FLAG_OWN_REALTY char(1) NOT NULL,
  CNT_CHILDREN smallint NOT NULL,
  AMT_INCOME_TOTAL real NOT NULL,
  AMT_CREDIT real NOT NULL,
  AMT_ANNUITY real NULL,
  AMT_GOODS_PRICE real NULL,
  NAME_TYPE_SUITE varchar(15) NULL,
  NAME_INCOME_TYPE varchar(20) NOT NULL,
  NAME_EDUCATION_TYPE varchar(29) NOT NULL,
  NAME_FAMILY_STATUS varchar(20) NOT NULL,
  NAME_HOUSING_TYPE varchar(19) NOT NULL,
  REGION_POPULATION_RELATIVE real NOT NULL,
  DAYS_BIRTH smallint NOT NULL,
  DAYS_EMPLOYED integer NOT NULL,
  DAYS_REGISTRATION real NOT NULL,
  DAYS_ID_PUBLISH smallint NOT NULL,
  OWN_CAR_AGE real NULL,
  FLAG_MOBIL smallint NOT NULL,
  FLAG_EMP_PHONE smallint NOT NULL,
  FLAG_WORK_PHONE smallint NOT NULL,
  FLAG_CONT_MOBILE smallint NOT NULL,
  FLAG_PHONE smallint NOT NULL,
  FLAG_EMAIL smallint NOT NULL,
  OCCUPATION_TYPE varchar(21) NULL,
  CNT_FAM_MEMBERS real NULL,
  REGION_RATING_CLIENT smallint NOT NULL,
  REGION_RATING_CLIENT_W_CITY smallint NOT NULL,
  WEEKDAY_APPR_PROCESS_START varchar(9) NOT NULL,
  HOUR_APPR_PROCESS_START smallint NOT NULL,
  REG_REGION_NOT_LIVE_REGION smallint NOT NULL,
  REG_REGION_NOT_WORK_REGION smallint NOT NULL,
  LIVE_REGION_NOT_WORK_REGION smallint NOT NULL,
  REG_CITY_NOT_LIVE_CITY smallint NOT NULL,
  REG_CITY_NOT_WORK_CITY smallint NOT NULL,
  LIVE_CITY_NOT_WORK_CITY smallint NOT NULL,
  ORGANIZATION_TYPE varchar(22) NOT NULL,
  EXT_SOURCE_1 real NULL,
  EXT_SOURCE_2 real NULL,
  EXT_SOURCE_3 real NULL,
  APARTMENTS_AVG real NULL,
  BASEMENTAREA_AVG real NULL,
  YEARS_BEGINEXPLUATATION_AVG real NULL,
  YEARS_BUILD_AVG real NULL,
  COMMONAREA_AVG real NULL,
  ELEVATORS_AVG real NULL,
  ENTRANCES_AVG real NULL,
  FLOORSMAX_AVG real NULL,
  FLOORSMIN_AVG real NULL,
  LANDAREA_AVG real NULL,
  LIVINGAPARTMENTS_AVG real NULL,
  LIVINGAREA_AVG real NULL,
  NONLIVINGAPARTMENTS_AVG real NULL,
  NONLIVINGAREA_AVG real NULL,
  APARTMENTS_MODE real NULL,
  BASEMENTAREA_MODE real NULL,
  YEARS_BEGINEXPLUATATION_MODE real NULL,
  YEARS_BUILD_MODE real NULL,
  COMMONAREA_MODE real NULL,
  ELEVATORS_MODE real NULL,
  ENTRANCES_MODE real NULL,
  FLOORSMAX_MODE real NULL,
  FLOORSMIN_MODE real NULL,
  LANDAREA_MODE real NULL,
  LIVINGAPARTMENTS_MODE real NULL,
  LIVINGAREA_MODE real NULL,
  NONLIVINGAPARTMENTS_MODE real NULL,
  NONLIVINGAREA_MODE real NULL,
  APARTMENTS_MEDI real NULL,
  BASEMENTAREA_MEDI real NULL,
  YEARS_BEGINEXPLUATATION_MEDI real NULL,
  YEARS_BUILD_MEDI real NULL,
  COMMONAREA_MEDI real NULL,
  ELEVATORS_MEDI real NULL,
  ENTRANCES_MEDI real NULL,
  FLOORSMAX_MEDI real NULL,
  FLOORSMIN_MEDI real NULL,
  LANDAREA_MEDI real NULL,
  LIVINGAPARTMENTS_MEDI real NULL,
  LIVINGAREA_MEDI real NULL,
  NONLIVINGAPARTMENTS_MEDI real NULL,
  NONLIVINGAREA_MEDI real NULL,
  FONDKAPREMONT_MODE varchar(21) NULL,
  HOUSETYPE_MODE varchar(16) NULL,
  TOTALAREA_MODE real NULL,
  WALLSMATERIAL_MODE varchar(12) NULL,
  EMERGENCYSTATE_MODE varchar(3) NULL,
  OBS_30_CNT_SOCIAL_CIRCLE real NULL,
  DEF_30_CNT_SOCIAL_CIRCLE real NULL,
  OBS_60_CNT_SOCIAL_CIRCLE real NULL,
  DEF_60_CNT_SOCIAL_CIRCLE real NULL,
  DAYS_LAST_PHONE_CHANGE real NULL,
  FLAG_DOCUMENT_2 smallint NOT NULL,
  FLAG_DOCUMENT_3 smallint NOT NULL,
  FLAG_DOCUMENT_4 smallint NOT NULL,
  FLAG_DOCUMENT_5 smallint NOT NULL,
  FLAG_DOCUMENT_6 smallint NOT NULL,
  FLAG_DOCUMENT_7 smallint NOT NULL,
  FLAG_DOCUMENT_8 smallint NOT NULL,
  FLAG_DOCUMENT_9 smallint NOT NULL,
  FLAG_DOCUMENT_10 smallint NOT NULL,
  FLAG_DOCUMENT_11 smallint NOT NULL,
  FLAG_DOCUMENT_12 smallint NOT NULL,
  FLAG_DOCUMENT_13 smallint NOT NULL,
  FLAG_DOCUMENT_14 smallint NOT NULL,
  FLAG_DOCUMENT_15 smallint NOT NULL,
  FLAG_DOCUMENT_16 smallint NOT NULL,
  FLAG_DOCUMENT_17 smallint NOT NULL,
  FLAG_DOCUMENT_18 smallint NOT NULL,
  FLAG_DOCUMENT_19 smallint NOT NULL,
  FLAG_DOCUMENT_20 smallint NOT NULL,
  FLAG_DOCUMENT_21 smallint NOT NULL,
  AMT_REQ_CREDIT_BUREAU_HOUR real NULL,
  AMT_REQ_CREDIT_BUREAU_DAY real NULL,
  AMT_REQ_CREDIT_BUREAU_WEEK real NULL,
  AMT_REQ_CREDIT_BUREAU_MON real NULL,
  AMT_REQ_CREDIT_BUREAU_QRT real NULL,
  AMT_REQ_CREDIT_BUREAU_YEAR real NULL);

CREATE TABLE bureau (
  SK_ID_CURR integer NOT NULL,
  SK_ID_BUREAU integer NOT NULL,
  CREDIT_ACTIVE varchar(8) NOT NULL,
  CREDIT_CURRENCY char(10) NOT NULL,
  DAYS_CREDIT smallint NOT NULL,
  CREDIT_DAY_OVERDUE smallint NOT NULL,
  DAYS_CREDIT_ENDDATE real NULL,
  DAYS_ENDDATE_FACT real NULL,
  AMT_CREDIT_MAX_OVERDUE real NULL,
  CNT_CREDIT_PROLONG smallint NOT NULL,
  AMT_CREDIT_SUM real NULL,
  AMT_CREDIT_SUM_DEBT real NULL,
  AMT_CREDIT_SUM_LIMIT real NULL,
  AMT_CREDIT_SUM_OVERDUE real NOT NULL,
  CREDIT_TYPE varchar(44) NOT NULL,
  DAYS_CREDIT_UPDATE integer NOT NULL,
  AMT_ANNUITY real NULL);

CREATE TABLE bureau_balance (
  SK_ID_BUREAU integer NOT NULL,
  MONTHS_BALANCE smallint NOT NULL,
  STATUS char(1) NOT NULL);

CREATE TABLE credit_card_balance (
  SK_ID_PREV integer NOT NULL,
  SK_ID_CURR integer NOT NULL,
  MONTHS_BALANCE smallint NOT NULL,
  AMT_BALANCE real NOT NULL,
  AMT_CREDIT_LIMIT_ACTUAL integer NOT NULL,
  AMT_DRAWINGS_ATM_CURRENT real NULL,
  AMT_DRAWINGS_CURRENT real NOT NULL,
  AMT_DRAWINGS_OTHER_CURRENT real NULL,
  AMT_DRAWINGS_POS_CURRENT real NULL,
  AMT_INST_MIN_REGULARITY real NULL,
  AMT_PAYMENT_CURRENT real NULL,
  AMT_PAYMENT_TOTAL_CURRENT real NOT NULL,
  AMT_RECEIVABLE_PRINCIPAL real NOT NULL,
  AMT_RECIVABLE real NOT NULL,
  AMT_TOTAL_RECEIVABLE real NOT NULL,
  CNT_DRAWINGS_ATM_CURRENT real NULL,
  CNT_DRAWINGS_CURRENT smallint NOT NULL,
  CNT_DRAWINGS_OTHER_CURRENT real NULL,
  CNT_DRAWINGS_POS_CURRENT real NULL,
  CNT_INSTALMENT_MATURE_CUM real NULL,
  NAME_CONTRACT_STATUS varchar(13) NOT NULL,
  SK_DPD smallint NOT NULL,
  SK_DPD_DEF smallint NOT NULL);

CREATE TABLE installments_payments (
  SK_ID_PREV integer NOT NULL,
  SK_ID_CURR integer NOT NULL,
  NUM_INSTALMENT_VERSION real NOT NULL,
  NUM_INSTALMENT_NUMBER smallint NOT NULL,
  DAYS_INSTALMENT real NOT NULL,
  DAYS_ENTRY_PAYMENT real NULL,
  AMT_INSTALMENT real NOT NULL,
  AMT_PAYMENT real NULL);

CREATE TABLE previous_application (
  SK_ID_PREV integer NOT NULL,
  SK_ID_CURR integer NOT NULL,
  NAME_CONTRACT_TYPE varchar(15) NOT NULL,
  AMT_ANNUITY real NULL,
  AMT_APPLICATION real NOT NULL,
  AMT_CREDIT real NULL,
  AMT_DOWN_PAYMENT real NULL,
  AMT_GOODS_PRICE real NULL,
  WEEKDAY_APPR_PROCESS_START varchar(9) NOT NULL,
  HOUR_APPR_PROCESS_START smallint NOT NULL,
  FLAG_LAST_APPL_PER_CONTRACT char(1) NOT NULL,
  NFLAG_LAST_APPL_IN_DAY smallint NOT NULL,
  RATE_DOWN_PAYMENT real NULL,
  RATE_INTEREST_PRIMARY real NULL,
  RATE_INTEREST_PRIVILEGED real NULL,
  NAME_CASH_LOAN_PURPOSE varchar(32) NOT NULL,
  NAME_CONTRACT_STATUS varchar(12) NOT NULL,
  DAYS_DECISION smallint NOT NULL,
  NAME_PAYMENT_TYPE varchar(41) NOT NULL,
  CODE_REJECT_REASON varchar(6) NOT NULL,
  NAME_TYPE_SUITE varchar(15) NULL,
  NAME_CLIENT_TYPE varchar(9) NOT NULL,
  NAME_GOODS_CATEGORY varchar(24) NOT NULL,
  NAME_PORTFOLIO varchar(5) NOT NULL,
  NAME_PRODUCT_TYPE varchar(7) NOT NULL,
  CHANNEL_TYPE varchar(26) NOT NULL,
  SELLERPLACE_AREA integer NOT NULL,
  NAME_SELLER_INDUSTRY varchar(20) NOT NULL,
  CNT_PAYMENT real NULL,
  NAME_YIELD_GROUP varchar(10) NOT NULL,
  PRODUCT_COMBINATION varchar(30) NULL,
  DAYS_FIRST_DRAWING real NULL,
  DAYS_FIRST_DUE real NULL,
  DAYS_LAST_DUE_1ST_VERSION real NULL,
  DAYS_LAST_DUE real NULL,
  DAYS_TERMINATION real NULL,
  NFLAG_INSURED_ON_APPROVAL real NULL);

