import pickle
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify

MODEL = lgb.Booster(model_file='lgbm_model.txt')
app = Flask(__name__)
app.secret_key = 'something_secret'

CAT_FEATURES = ['account_status', 'account_worst_status_0_3m', 'account_worst_status_12_24m', 
                'account_worst_status_3_6m', 'account_worst_status_6_12m', 'merchant_category', 
                'merchant_group', 'name_in_email', 'status_last_archived_0_24m', 
                'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m', 
                'status_max_archived_0_6_months', 'status_max_archived_0_12_months', 
                'status_max_archived_0_24_months', 'worst_status_active_inv', 'has_paid']

REST_FEATURES = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m', 'account_days_in_rem_12_24m',
                 'account_days_in_term_12_24m', 'account_incoming_debt_vs_paid_0_24m', 'age', 
                 'avg_payment_span_0_12m', 'avg_payment_span_0_3m', 'max_paid_inv_0_12m', 'max_paid_inv_0_24m',
                 'num_active_div_by_paid_inv_0_12m', 'num_active_inv', 'num_arch_dc_0_12m', 'num_arch_dc_12_24m',
                 'num_arch_ok_0_12m', 'num_arch_ok_12_24m', 'num_arch_rem_0_12m',
                 'num_arch_written_off_0_12m', 'num_arch_written_off_12_24m', 'num_unpaid_bills',
                 'recovery_debt', 'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
                 'sum_paid_inv_0_12m', 'time_hours']

@app.route('/')
def home_endpoint():
    return 'Study case REST API, Vadim Kuzmin'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = pd.DataFrame(data)
        if 'uuid' in data.columns:
            uuid = data.uuid.values
            data.drop('uuid', 1, inplace=True)
        else:
            uuid = None
        if 'default' in data.columns:
            data.drop('default', 1, inplace=True)
        for cat_feature in CAT_FEATURES:
            data[cat_feature] = data[cat_feature].astype('category')
        for feature in REST_FEATURES:
            data[feature] = data[feature].astype('float')
        prediction = tuple(MODEL.predict(data))
        if uuid is not None:
            return jsonify(dict(zip(uuid, prediction)))
    return jsonify({'preds': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)