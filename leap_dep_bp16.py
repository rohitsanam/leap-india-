#*********************************** LIBRARIES *********************************#
import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st 
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#import seaborn as sns
from PIL import Image
from sklearn.impute import KNNImputer
from feature_engine.outliers import Winsorizer
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from pathlib import Path
import base64
from PIL import Image
# from pmdarima import auto_arima
#*********************************** MAPE FUNCTION*********************************#
def MAPE(org,pred):
    temp = np.abs((org-pred)/org)*100
    return np.mean(temp)
# clean_customer_name
def clean_customer_name(text):
    specified_words = ['Private Limited','Limited', 'Pvt', 'Pvt.', 'Private']
    # Split the text into words
    words = text.split()

    # Find the indices of the specified words (case-insensitive)
    indices_to_remove = set()
    for word_to_remove in specified_words:
        for i, word in enumerate(words):
            if word.lower() == word_to_remove.lower():
                indices_to_remove.add(i)
                break

    # If specified words are found, remove all words from those indices onward
    if indices_to_remove:
        min_index = min(indices_to_remove)
        words = words[:min_index]

    # Join the remaining words back into a string
    result = ' '.join(words)

    return result
# Define a function to resample and save each dataset
def resample_and_save(dataset_name, dataset):
    # Rename the 'QTY' column to 'Quantity'
    dataset.rename(columns={'QTY': 'Quantity'}, inplace=True)

    # Convert the 'Date' column to a datetime object if it's not already
    dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%Y')

    # Set the 'Date' column as the index
    dataset.set_index('Date', inplace=True)

    # Resample the dataset with a weekly frequency and sum the 'Quantity' column
    resampled_dataset = dataset[['Quantity']].resample('W').sum()

    # Rename the dataset with "_W" at the end
    new_dataset_name = dataset_name + '_W'

    # Save the resampled dataset
    globals()[new_dataset_name] = resampled_dataset
    return resampled_dataset
#*********************************** Main function ********************************#
def main():
    # Customize the Streamlit theme
    st.set_page_config(
        page_title="Forecasting Streamlit App",
        page_icon="🔮",  # Unicode symbol for forecasting

    )
 
    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='width: 100px; height: auto;'>".format(
        img_to_bytes("leap.jpeg")
    )
    header2_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='width: 150px; height: auto;'>".format(
        img_to_bytes("innodatatics-white1.png")
    )

    # Streamlit app
    #st.image(logo_image, width=200)

    title = "Wooden Pallet Forecasting"
    padding_length = len(title) * 2

    html_temp = f"""
        <div style="background: linear-gradient(90deg, #87CEEB, #1E90FF); border-radius: 25px; padding: 10px {padding_length}px 10px 10px; color: black; text-align: center; margin: 10px 0px; box-shadow: 5px 5px 15px 5px #000000;">
            <p style="font-size: 25px; font-weight: bold;">&nbsp;{header_html}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{title}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{header2_html}</p>
        </div>
    """
    # html_temp = f"""
    #     <div style="background: linear-gradient(90deg, #87CEEB, #1E90FF); border-radius: 25px; padding: 10px {padding_length}px 10px 10px; color: black; text-align: center; margin: 10px 0px; box-shadow: 5px 5px 15px 5px #000000;">
    #         <p style="font-size: 30px; font-weight: bold;">&nbsp;{header_html}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{title}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{header2_html}</p>
    #     </div>
    # """
    st.markdown(html_temp, unsafe_allow_html=True)
    input_style = """
        <style>
        
        [data-baseweb="select"],
        [data-testid="stFileUploadDropzone"],
        [role="option"]
        [data-baseweb="slider"],
        [role="radiogroup"],
        [data-testid="baseButton-secondary"]{
            background: linear-gradient(90deg, #87CEEB, #1E90FF);
            color: black;
            padding: 5px;
            border: 1px solid #000000;
            border-radius: 25px
        }

        
    </style>
    """

    # Display custom CSS style
    st.markdown(input_style, unsafe_allow_html=True)
    # st.sidebar.title("Forecasting")
    
    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    
    st.text("")
    # data = pd.DataFrame()
    uploadedFile = st.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    # Check if a file has been uploaded
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)
            # Continue with your data processing logic here
            st.write("File uploaded and data processed.")
        except Exception as csv_error:
            try:
                data = pd.read_excel(uploadedFile)
                # Continue with your data processing logic here
                st.write("File uploaded and data processed.")
            except Exception as excel_error:
                st.error(f"Error loading file: {csv_error}\n{excel_error}")
    else:
        # Display a message or loading indicator while waiting for user input
        st.warning("You need to upload a csv or excel format file only.")
        return
    if data is not None:        
        data1 = data.copy(deep = True)    
        # Separate DataFrames for 'Allot' and 'Return'
        allot_df = data1[data1['Model 2'] == 'Allot']
        return_df = data1[data1['Model 2'] == 'Return']
        allot_df = allot_df[['Date', 'Customer Name','City','QTY']]
        
        # Apply the cleaning function to the 'Customer Name' column
        allot_df['Customer_New_Name'] = allot_df['Customer Name'].apply(clean_customer_name)
        # Assuming you have a Pandas Series named 'my_series'
        # companies = allot_df['Customer_New_Name'].value_counts().index[:6].tolist()
        selected_Companies = st.selectbox("Select an Industry", ['Beverages (Nov-May)','E-commerce(Jun-Nov)','Generic(All season)'])
        company_names = [
            "Hindustan Coca Cola Beverages Private Limited _  Goblej Plant _ HMA1",
            "Hindustan Coca Cola Beverages Private Limited _Ahmedabad Sanand_HMAH",
            "Alpla India Private Limited _ Sangareddy",
            "Epitome Petropack Limited _ Kolkata",
            "Hindustan Coca Cola Beverages Private Limited _Khurda _ HMF1",
            "Hindustan Coca Cola Beverages Pvt Ltd -GAPL - Howrah",
            "Hindustan Coca Cola Beverages Private Limited _ Bidadi Greenfield_HMKJ",
            "Hindustan Coca Cola Beverages Private Limited_Kanchenkanya_HMS3",
            "Babri Polypet Private Limited_ Haridwar",
            "SLMG Beverages Pvt. Limited _ Lucknow",
            "Oricon Enterprises Limited _ Khordha",
            "Hindustan Coca Cola Beverages Private Limited _ Bangalore_ HCKB",
            "Moon Beverages Limited _ Dasna"
            "Pepsico India Holdings Private Limited_Ludhiana",
            "Pepsico India Holdings Pvt. Ltd. _ Patiala _Channo",
            "Alpla India Private Limited _ Pune",
            "Cans and  Closures Pvt. Ltd. _ Hooghly"
        ]
        company_names1 = [
            "Manjushree Technopack Limited _ Bangalore _Bidadi",
            "Chemco Plastic Industries Pvt. Ltd. _ Vadodara",
            "Bericap India Private Limited_Pune",
            "Manjushree Technopack Limited_Rudrapur"
        ]
        company_names2 = [
            "Amazon Seller Services Pvt. Ltd. _ Bangalore_BLR4",
            "Amazon Seller Services Pvt. Ltd. - Gurgaon_ DED3"
        ]
        if selected_Companies=='Beverages (Nov-May)':
            selected_category = st.selectbox("Select a Customer", company_names )
        
        elif selected_Companies == 'Generic(All season)':
              selected_category = st.selectbox("Select a Customer", company_names1 )
              
        elif selected_Companies == 'E-commerce(Jun-Nov)':
              selected_category = st.selectbox("Select a Customer", company_names2 )
    
        #     selected_category = st.selectbox("Select a Customer", ['Manjushree Technopack'])
        
        words = selected_category.replace("_", " ").replace("-", " ").split()
        # Create the dataset name by joining the first letter of each word
        dataset_name = ''.join([word[0] for word in words])
        # Filter the DataFrame based on the selected company name
        finaldata = allot_df[allot_df['Customer Name'] == selected_category]
       
        
        # st.table(filtered_df) 
        # Loop through each dataset and resample it
        
        dataset = resample_and_save(dataset_name, finaldata)
        if selected_Companies=='Beverages (Nov-May)':
            mask = (dataset.index.month.isin([11, 12, 1, 2, 3, 4, 5])) & (dataset.index.year >= 2021)
            dataset = dataset[mask]
            desired_date = '2021-11-07'
            
            # Filter the dataset for the desired date
            dataset = dataset[dataset.index >= desired_date]
            # Replace zeros with NaN
            # st.write(dataset)
            # zeros_count = (dataset['Quantity'] == 0).sum()
            # Calculate the percentage of zeros in each column
            zero_percentages = int((dataset['Quantity'] == 0).mean() * 100)
    
            st.write(f'Percentage of zero: {zero_percentages}%')
            dataset['Quantity'].replace(0, np.nan, inplace=True)
            # st.table(dataset)

            
        elif selected_Companies=='E-commerce(Jun-Nov)':
            mask = (dataset.index.month.isin([6, 7, 8, 9, 10,11])) & (dataset.index.year >= 2021)
            dataset = dataset[mask]
            desired_date = '2021-11-07'
            
            # Filter the dataset for the desired date
            dataset = dataset[dataset.index >= desired_date]
            # Replace zeros with NaN
            # st.write(dataset)
            # zeros_count = (dataset['Quantity'] == 0).sum()
            # Calculate the percentage of zeros in each column
            zero_percentages = int((dataset['Quantity'] == 0).mean() * 100)
    
            st.write(f'Percentage of zero: {zero_percentages}%')
            dataset['Quantity'].replace(0, np.nan, inplace=True)

        elif selected_Companies=='Generic(All season)':
            dataset['Quantity'].replace(0, np.nan, inplace=True)
            zero_percentages = int((dataset['Quantity'].fillna(0) == 0).mean() * 100)
    
            st.write(f'Percentage of zero: {zero_percentages}%')
            
        # KNN imputation using sklearn's KNNImputer
        imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed
        clean_dataset = dataset.copy()
        clean_dataset['Quantity'] = imputer.fit_transform(clean_dataset[['Quantity']])
        # st.table(clean_dataset.head(20))
        # dataset = globals()[dataset_name]
        try:
            winsorizer = Winsorizer(
                    capping_method='iqr',  # You can use 'iqr' for interquartile range-based winsorization
                    tail='both',           # Winsorize both lower and upper tails
                    fold=1.5               # Adjust the fold parameter as needed
                )
            # Perform Winsorization on the 'Quantity' column
            clean_dataset['Quantity'] = winsorizer.fit_transform(clean_dataset[['Quantity']])
            
            # globals()[winsorized_dataset_name] = dataset
            
        except Exception as e:
            st.write(f"Error processing dataset: {dataset_name}")
            st.error(f"Error message: {str(e)}")
        target = clean_dataset['Quantity']
        # Models to apply
        # Models to apply
        models = [
            ('Linear_Model', 'Quantity ~ t'),
            ('Exponential_Model', 'np.log(Quantity) ~ t'),
            ('Quadratic_Model', 'Quantity ~ t + np.power(t, 2)'),
            ('Add_seasonality_Model', 'Quantity ~ np.sin(t) + np.cos(t)'),
            ('Mul_seasonality_Model', 'np.log(Quantity) ~ np.sin(t) + np.cos(t)'),
            ('ASQT_Model', 'Quantity ~ t + np.power(t, 2) + np.sin(t) + np.cos(t)'),
            ('ASLinT_Model', 'Quantity ~ t + np.sin(t) + np.cos(t)'),
            ('MSLinT_Model', 'np.log(Quantity) ~ t + np.sin(t) + np.cos(t)'),
            ('MSQT_Model', 'np.log(Quantity) ~ t + np.power(t, 2) + np.sin(t) + np.cos(t)'),
            ('Holt_Model', None),
            ('Holts_Winter_Add_Seasonality_Model', None),
            ('Holts_Winter_Mul_Seasonality_Model', None)
            
        ]
        # 
        # Create an empty dictionary to store the results
        # result_data = {}
        dataset_df = clean_dataset.copy()
        dataset_df2 = clean_dataset.copy()
        dataset_df2 = dataset_df2.reset_index()
        test_size1 = int(len(dataset_df2) * 0.20)
        train_df1= dataset_df2.iloc[:-test_size1]
        test_df1 = dataset_df2.iloc[-test_size1:]
        # Assuming you have a single dataset called 'dataset_df'
        dataset_df["t"] = np.arange(1, len(dataset_df) + 1)
        test_size = int(len(dataset_df) * 0.20)
        train_df = dataset_df.iloc[:-test_size]
        test_df = dataset_df.iloc[-test_size:]
        # st.write(train_df)
        # Create a dictionary to store model results for this dataset
        dataset_results = {}
        trainpred = {}
        testpred = {}
        # Loop through each model
        for model_name, formula in models:
            try:
                if formula:
                    if 'np.log' in formula:                    
                        # If formula contains log transformation, apply np.exp on predictions
                        # transformed_formula = formula.replace('np.log', 'np.exp')
                        model = smf.ols(formula, data=train_df).fit()
                        pred_train = np.exp(model.predict(train_df))
                        pred_test = np.exp(model.predict(test_df))
                    else:
                        model = smf.ols(formula, data=train_df).fit()
                        pred_train = model.predict(train_df)
                        pred_test = model.predict(test_df)
                else:
                    
                    if model_name == 'SES_Model':
                        model = SimpleExpSmoothing(train_df1['Quantity'].values).fit(smoothing_level=0.2,)
                        pred_train = model.predict(start=train_df1.index[0], end=train_df1.index[-1])
                        pred_test = model.predict(start=test_df1.index[0], end=test_df1.index[-1])
                    elif model_name == 'Holt_Model':
                        model = Holt(train_df1['Quantity'].values).fit()
                        pred_train = model.predict(start=train_df1.index[0], end=train_df1.index[-1])
                        pred_test = model.predict(start=test_df1.index[0], end=test_df1.index[-1])
                    elif model_name == 'Holts_Winter_Add_Seasonality_Model':
                        model = ExponentialSmoothing(train_df1['Quantity'].values, seasonal="add", trend="add", seasonal_periods=7).fit()
                        pred_train = model.predict(start=train_df1.index[0], end=train_df1.index[-1])
                        pred_test = model.predict(start=test_df1.index[0], end=test_df1.index[-1])
                        
                    elif model_name == 'Holts_Winter_Mul_Seasonality_Model':
                        model = ExponentialSmoothing(train_df1['Quantity'].values, seasonal="mul", trend="add", seasonal_periods=7).fit()
                        pred_train = model.predict(start=train_df1.index[0], end=train_df1.index[-1])
                        pred_test = model.predict(start=test_df1.index[0], end=test_df1.index[-1])
                        
                # Store the predictions in the respective dictionaries
                trainpred[model_name] = pred_train
                testpred[model_name] = pred_test
                # Calculate Train MAPE and Test MAPE
                mape_train = MAPE(train_df['Quantity'], pred_train)
                mape_test = MAPE(test_df['Quantity'], pred_test)
    
                # Store the results in the dataset_results dictionary
                dataset_results[model_name + '_train'] = mape_train
                dataset_results[model_name + '_test'] = mape_test
                        # if 'level_0' in dataset_df2.columns:
                        #     dataset_df2.drop(columns=['level_0'], inplace=True)
                        # dataset_df2 = dataset_df2.reset_index(drop=True)
                        # test_size1 = int(len(dataset_df2) * 0.20)
                        # train_df1= dataset_df2.iloc[:-test_size1]
                        # test_df1 = dataset_df2.iloc[-test_size1:]
            except Exception as e:
                st.error(f"Error with model '{model_name}': {str(e)}")
        trainpred_df = pd.DataFrame(trainpred)
        testpred_df = pd.DataFrame(testpred)
        col1, col2 = st.columns(2)
        # with col1:
        #     st.write('train results')
        #     st.write(trainpred_df)
        # with col2:
        #     st.write('test results')
        #     st.write(testpred_df)
        # Convert the dataset_results dictionary into a DataFrame
        result_df = pd.DataFrame(dataset_results, index=[0])
        # Sort the result_df DataFrame in ascending order based on the model name
        # result_df = result_df.reindex(sorted(result_df.columns), axis=1)
        combined_mape_dict = {}
        
        for column in result_df.columns:
            if column.endswith("_train"):
                model_name = column.replace("_train", "")
                combined_mape = result_df[column] + result_df[model_name + "_test"]
                combined_mape_dict[model_name] = combined_mape.values[0]
        
        # Sort the dictionary by combined MAPE in ascending order
        sorted_combined_mape_dict = dict(sorted(combined_mape_dict.items(), key=lambda x: x[1]))
        
        # Rearrange the DataFrame based on the sorted dictionary
        sorted_columns = []
        for key in sorted_combined_mape_dict.keys():
            train_column = f"{key}_train"
            test_column = f"{key}_test"
            sorted_columns.extend([train_column, test_column])
        
        sorted_result_df = result_df[sorted_columns]
        
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        # st.table(sorted_result_df.iloc[:, :8].style.background_gradient(cmap = cm))
        column_names = list(sorted_result_df.columns)
        
        train_list = [col.replace('_train', '') for col in column_names if col.endswith('_train')]
        test_list = [col.replace('_test', '') for col in column_names if col.endswith('_test')]
        # Remove 'SES_Model' if it exists in the train_list
        train_list = [model for model in train_list if model != 'SES_Model'][:4]
        test_list = [model for model in test_list if model != 'SES_Model'][:4]
        
        # hard_list = ['Holts_Winter_Add_Seasonality_Model', 'Holts_Winter_Mul_Seasonality_Model', 'Holt_Model']
        ensemble_train = trainpred_df.loc[:, train_list]
        ensemble_test = testpred_df.loc[:, train_list]
        
        ensemble_train['target'] = train_df['Quantity'].values
        ensemble_test['target'] = test_df['Quantity'].values
        # col5, col6 = st.columns(2)
        # with col5:
        #     st.write(ensemble_train)
        # with col6:
        #     st.write(ensemble_test)
        selected_model = train_list[0]
        trainmape=sorted_result_df.iloc[0, 0]
        testmape=sorted_result_df.iloc[0, 1]
        
        col7, col8 = st.columns(2)
        with col7:
            st.write(f'<span style="font-weight:bold">Best individual Model</span>', unsafe_allow_html=True)
            st.write(f'Train MAPE for {selected_model}: {int(trainmape)}%')
            st.write(f'Train MAPE for {selected_model}: {int(testmape)}%')
            st.write('')
       
            
        ensemble = RandomForestRegressor(n_estimators=12,max_depth=3,min_samples_split=2,min_samples_leaf=2,random_state=42)
        # from sklearn.ensemble import GradientBoostingRegressor
        # ensemble = GradientBoostingRegressor(alpha=0.95)
        my_rf = ensemble.fit(ensemble_train.drop(['target'], axis = 1), ensemble_train['target'])
        # predict on the Train period
        preds_rf_train = my_rf.predict(ensemble_train.drop(['target'], axis = 1))
        preds_rf_test = my_rf.predict(ensemble_test.drop(['target'], axis = 1))

        # MAPE on RF model
        mape_train_rf = MAPE(ensemble_train['target'], preds_rf_train)
        mape_test_rf = MAPE(ensemble_test['target'], preds_rf_test)
        # Reset the index for both DataFrames
        with col8:
            st.write(f'<span style="font-weight:bold">Ensemble Model</span> ', unsafe_allow_html=True)
            st.write(f'Train MAPE for Ensemble: {int(mape_train_rf)}%')
            st.write(f'Test MAPE for Ensemble: {int(mape_test_rf)}%')
            
        # Train the RF model
        # Map items from new_models to corresponding tuples from models
        mapped_models = [model for model in models if model[0] in train_list]
        
        list1 = [model_name for model_name, _ in models]
        # list1.insert(0, 'Please select model')
        # list1.append('Ensemble')
        # Dropdown to select a model

        if selected_Companies=='Beverages (Nov-May)':
            num_periods = st.slider("Select Number of Weeks for Forecasting(Nov-May)", min_value=1, max_value=24, value=7)
        elif selected_Companies== "E-commerce(Jun-Nov)":
            num_periods = st.slider("Select Number of Weeks for Forecasting", min_value=1, max_value=24, value=7)            
        elif selected_Companies== "Generic(All season)":
            num_periods = st.slider("Select Number of Weeks for Forecasting(Nov-May)", min_value=1, max_value=24, value=7)
        st.write(f'<span style="font-weight:bold">Note: Apply a 90% confidence interval if the zero percentage is higher than 35%.</span> ', unsafe_allow_html=True)

        
        # Get the selected error percentage from the user using a radio button
        selected_error_percentage = st.radio("Select Confidence Interval", [95, 90])
        
        if selected_error_percentage == 95:
            error_percentage = 5
        else:
            error_percentage = 10    
            
        # future_preds = 0 
        # Initialize future_preds
        future_preds = None
        if selected_model == 'Please select model':
            st.write('Please select model')
        else:
            # Assuming you have a single dataset called 'dataset_df'
            dataset_df = clean_dataset.copy()
            dataset_df["t"] = np.arange(1, len(dataset_df) + 1)
            # Slider to select the number of weeks for forecasting
            # Use the slider
            
            # num_periods = st.slider("Select Number of Weeks for Forecasting", min_value=1, max_value=52, value=10)
            # Perform modeling and forecasting based on the selected model
            for model_name, formula in models:
                if model_name == selected_model:
                    try:
                        if formula:
                            if 'np.log' in formula: 
                                model = smf.ols(formula, data=dataset_df).fit()
                                
                                # Forecasting for the future
                                # Define the number of future periods you want to forecast
                                  # Adjust as needed
                                future_t = np.arange(len(dataset_df) + 1, len(dataset_df) + num_periods + 1)
                                future_data = pd.DataFrame({'t': future_t})
                                future_preds = np.exp(model.predict(future_data))
                            else:
                            
                                # Fit the model with the specified formula
                                model = smf.ols(formula, data=dataset_df).fit()
                                
                                # Forecasting for the future
                                # Define the number of future periods you want to forecast
                                  # Adjust as needed
                                future_t = np.arange(len(dataset_df) + 1, len(dataset_df) + num_periods + 1)
                                future_data = pd.DataFrame({'t': future_t})
                                future_preds = model.predict(future_data)
                        else:
                            
                            # Handle models that don't require a formula for forecasting
                           
                            # Handle models that don't require a formula for forecasting
                            if model_name == 'SES_Model':
                                future_model = SimpleExpSmoothing(dataset_df['Quantity']).fit()
                                future_preds = future_model.forecast(steps=num_periods)
                            elif model_name == 'Holt_Model':
                                future_model = Holt(dataset_df['Quantity']).fit()
                                future_preds = future_model.forecast(steps=num_periods)
                            elif model_name == 'Holts_Winter_Add_Seasonality_Model':
                                future_model = ExponentialSmoothing(dataset_df['Quantity'], seasonal="add", trend="add", seasonal_periods=7).fit()
                                future_preds = future_model.forecast(steps=num_periods)
                            elif model_name == 'Holts_Winter_Mul_Seasonality_Model':
                                future_model = ExponentialSmoothing(dataset_df['Quantity'], seasonal="mul", trend="add", seasonal_periods=7).fit()
                                future_preds = future_model.forecast(steps=num_periods)

                    except Exception as e:
                        st.error(f"Error with model '{selected_model}': {str(e)}")
                        
        if future_preds is not None:
            # Display forecasting results
            st.subheader("Forecasting Results")
            st.write("Future Forecast:")
        else:
            st.error("No forecasting results available.")


        if selected_Companies=='Beverages (Nov-May)':
            last_date = dataset_df.index[-1]
            
            # months = [12, 1, 2, 3, 4, 5]  # December to May
            # year = last_date.year  # Year of the last date in your dataset
            # day = last_date.day
            # # Generate dates for the first week of each month within the desired range
            # future_dates = [pd.to_datetime(f'{year}-{month:02d}-{day}') + pd.DateOffset(7 * i) for month in months for i in range(num_periods)]
            # # Create a DataFrame for future predictions
            weekly_dates = []
            
            # Iterate through the date range in weekly increments, considering only dates in December to May
            current_date = last_date
            weeks_count = 0
            e_num = num_periods+1
            while weeks_count < e_num:
                # Check if the current date is within December to May
                if current_date.month in [11, 12, 1, 2, 3, 4, 5]:
                    weekly_dates.append(current_date.strftime('%Y-%m-%d'))
                    weeks_count += 1
            
                # Move to the next week
                current_date += timedelta(weeks=1)
            weekly_dates.pop(0) 
            # Print the weekly dates
            # for date in weekly_dates:
            #     print(date)
            future_preds_df = pd.DataFrame({'Date': weekly_dates,
                                            'Quantity': future_preds})
        if selected_Companies=='Beverages (Nov-May)':
            last_date = dataset_df.index[-1]
            
            # months = [12, 1, 2, 3, 4, 5]  # December to May
            # year = last_date.year  # Year of the last date in your dataset
            # day = last_date.day
            # # Generate dates for the first week of each month within the desired range
            # future_dates = [pd.to_datetime(f'{year}-{month:02d}-{day}') + pd.DateOffset(7 * i) for month in months for i in range(num_periods)]
            # # Create a DataFrame for future predictions
            weekly_dates = []
            
            # Iterate through the date range in weekly increments, considering only dates in December to May
            current_date = last_date
            weeks_count = 0
            e_num = num_periods+1
            while weeks_count < e_num:
                # Check if the current date is within December to May
                if current_date.month in [11, 12, 1, 2, 3, 4, 5]:
                    weekly_dates.append(current_date.strftime('%Y-%m-%d'))
                    weeks_count += 1
            
                # Move to the next week
                current_date += timedelta(weeks=1)
            weekly_dates.pop(0) 
            # Print the weekly dates
            # for date in weekly_dates:
            #     print(date)
            future_preds_df = pd.DataFrame({'Date': weekly_dates,
                                            'Quantity': future_preds})          
        elif selected_Companies=='E-commerce(Jun-Nov)':
            last_date = dataset_df.index[-1]
            
            # months = [12, 1, 2, 3, 4, 5]  # December to May
            # year = last_date.year  # Year of the last date in your dataset
            # day = last_date.day
            # # Generate dates for the first week of each month within the desired range
            # future_dates = [pd.to_datetime(f'{year}-{month:02d}-{day}') + pd.DateOffset(7 * i) for month in months for i in range(num_periods)]
            # # Create a DataFrame for future predictions
            weekly_dates = []
            
            # Iterate through the date range in weekly increments, considering only dates in December to May
            current_date = last_date
            weeks_count = 0
            e_num = num_periods+1
            while weeks_count < e_num:
                # Check if the current date is within December to May
                if current_date.month in [6, 7, 8, 9, 10,11]:
                    weekly_dates.append(current_date.strftime('%Y-%m-%d'))
                    weeks_count += 1
            
                # Move to the next week
                current_date += timedelta(weeks=1)
            weekly_dates.pop(0) 
            # Print the weekly dates
            # for date in weekly_dates:
            #     print(date)
            future_preds_df = pd.DataFrame({'Date': weekly_dates,
                                            'Quantity': future_preds})              
        else:
            future_preds_df = pd.DataFrame({'Date': pd.date_range(start=dataset_df.index[-1] + pd.DateOffset(7), periods=num_periods, freq='W'),
                                        'Quantity': future_preds})
                  
        # Reset the index to start from 0
        future_preds_df.reset_index(drop=True, inplace=True)
        future_preds_df['Quantity'] = future_preds_df['Quantity'].astype(int)
        
        if st.button('Individual Forecast'):
            col3, col4 = st.columns(2)
            with col3:
                
                    # st.write('forecasting....')
    
                    # result = predict(data, user, pw, db)
                    # Create a DataFrame for future predictions with "Date" and "Quantity" columns
                    # Generate dates for December to May (first week of each month)
                    
                    future_preds_df['Lower Bound'] = (future_preds_df['Quantity'] * (1 - error_percentage / 100)).astype(int)
                    future_preds_df['Upper Bound'] = (future_preds_df['Quantity'] * (1 + error_percentage / 100)).astype(int)
                    future_preds_df['Quantity'] = future_preds_df['Quantity'].astype(int)
                    st.write(future_preds_df)
                    
            with col4:
                import matplotlib.pyplot as plt
                # button_html = f'<button style="background-color: #e02d4b; color: white;">Plot</button>'
                # # Display the button using the HTML generated
                # button_clicked = st.markdown(button_html, unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(future_preds_df['Date'], future_preds_df['Quantity'].values, color='yellow', label='Prediction')
                #ax.plot(sarima_b.loc['2021-01-31':]['Forecast'], color = 'red', label='prediction')
                ax.fill_between(future_preds_df['Date'], future_preds_df['Lower Bound'].values, future_preds_df['Upper Bound'].values, color='lightgray', label='Confidence Interval')
                #plt.scatter(sarima_b.loc['2021-01-31':].index, sarima_b.loc['2021-01-31':]['Forecast'], color='red', marker='o', label='Highlighted Prediction')
                #ax.plot(new_forecast.index, new_forecast['Forecast'], color='green', linewidth=2, label='Forecast')
                plt.scatter(future_preds_df['Date'], future_preds_df['Quantity'].values, color='red', marker='o', label='Highlighted Prediction')
                plt.xticks(rotation=45)
                ax.set_xlabel('Date')
                ax.set_ylabel('Quantity')
                ax.set_title(f'Forecasting with {selected_error_percentage}% Confidence Interval')
                # Display the legend
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                
        if st.button('Ensemble Forecast'):
            
            forecast_dict = {}
            for model_name, formula in mapped_models:
                try:
                    if formula:
                        if 'np.log' in formula:
                            model = smf.ols(formula, data=dataset_df).fit()
                            future_t = np.arange(len(dataset_df) + 1, len(dataset_df) + num_periods + 1)
                            future_data = pd.DataFrame({'t': future_t})
                            future_preds = np.exp(model.predict(future_data))
                        else:
                            model = smf.ols(formula, data=dataset_df).fit()
                            future_t = np.arange(len(dataset_df) + 1, len(dataset_df) + num_periods + 1)
                            future_data = pd.DataFrame({'t': future_t})
                            future_preds = model.predict(future_data)
                    else:
                        if model_name == 'SES_Model':
                            future_model = SimpleExpSmoothing(dataset_df['Quantity']).fit()
                            future_preds = future_model.forecast(steps=num_periods)
                        elif model_name == 'Holt_Model':
                            future_model = Holt(dataset_df['Quantity']).fit()
                            future_preds = future_model.forecast(steps=num_periods)
                        elif model_name == 'Holts_Winter_Add_Seasonality_Model':
                            future_model = ExponentialSmoothing(dataset_df['Quantity'], seasonal="add", trend="add", seasonal_periods=7).fit()
                            future_preds = future_model.forecast(steps=num_periods)
                        elif model_name == 'Holts_Winter_Mul_Seasonality_Model':
                            future_model = ExponentialSmoothing(dataset_df['Quantity'], seasonal="mul", trend="add", seasonal_periods=7).fit()
                            future_preds = future_model.forecast(steps=num_periods)
                    
                    # Save the forecasts in the dictionary with the model name as the key
                    forecast_dict[model_name] = future_preds.values
                    
                except Exception as e:
                    print(f"Error with model '{model_name}': {str(e)}")
            
            # Create a DataFrame from the forecast dictionary
            forecast_df = pd.DataFrame(forecast_dict)
            forecast_df = forecast_df[train_list]
            ensemble_train.reset_index(drop=True, inplace=True)
            ensemble_test.reset_index(drop=True, inplace=True)
            
            # Concatenate horizontally
            entiredata = pd.concat([ensemble_train, ensemble_test], axis=0)
            # st.dataframe(entiredata)
            # ensemble = RandomForestRegressor(n_estimators=12,max_depth=3,min_samples_split=2,min_samples_leaf=2,random_state=42)
            new_my_rf = ensemble.fit(entiredata.drop(['target'], axis = 1), entiredata['target'])
            new_forecast = new_my_rf.predict(forecast_df)
            if selected_Companies=='Beverages (Nov-May)':
                future_preds_df1 = pd.DataFrame({'Date': weekly_dates,
                                                'Quantity': new_forecast})
            elif selected_Companies=='E-commerce(Jun-Nov)':
                future_preds_df1 = pd.DataFrame({'Date': weekly_dates,
                                                'Quantity': new_forecast})
            else:
                future_preds_df1 = pd.DataFrame({'Date': pd.date_range(start=dataset_df.index[-1] + pd.DateOffset(7), periods=num_periods, freq='W'),
                                            'Quantity': new_forecast})
            # st.write(future_preds_df1)
            # Display the DataFrame with model names as column names
            # st.write(forecast_df)
            col5, col6 = st.columns(2)
            with col5:
                
                    # st.write('forecasting....')
    
                    # result = predict(data, user, pw, db)
                    # Create a DataFrame for future predictions with "Date" and "Quantity" columns
                    # Generate dates for December to May (first week of each month)
                    future_preds_df1['Lower Bound'] = (future_preds_df1['Quantity'] * (1 - error_percentage/100)).astype(int)
                    future_preds_df1['Upper Bound'] = (future_preds_df1['Quantity'] * (1 + error_percentage/100)).astype(int)
                    future_preds_df1['Quantity'] = future_preds_df1['Quantity'].astype(int)
                    
                    st.write(future_preds_df1)
                    
            with col6:
                import matplotlib.pyplot as plt
                # button_html = f'<button style="background-color: #e02d4b; color: white;">Plot</button>'
                # # Display the button using the HTML generated
                # button_clicked = st.markdown(button_html, unsafe_allow_html=True)
            
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(future_preds_df1['Date'], future_preds_df1['Quantity'].values, color='yellow', label='Prediction')
                #ax.plot(sarima_b.loc['2021-01-31':]['Forecast'], color = 'red', label='prediction')
                ax.fill_between(future_preds_df1['Date'], future_preds_df1['Lower Bound'].values, future_preds_df1['Upper Bound'].values, color='lightgray', label='Confidence Interval')
                #plt.scatter(sarima_b.loc['2021-01-31':].index, sarima_b.loc['2021-01-31':]['Forecast'], color='red', marker='o', label='Highlighted Prediction')
                #ax.plot(new_forecast.index, new_forecast['Forecast'], color='green', linewidth=2, label='Forecast')
                plt.scatter(future_preds_df1['Date'], future_preds_df1['Quantity'].values, color='red', marker='o', label='Highlighted Prediction')
                plt.xticks(rotation=45)
                ax.set_xlabel('Date')
                ax.set_ylabel('Quantity')
                ax.set_title(f'Forecasting with {selected_error_percentage}% Confidence Interval')
                # Display the legend
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                # Display the DataFrame in an expander
                # with st.expander('View DataFrame'):
                #     st.write(future_preds_df)             
                
if __name__=='__main__':
    main()