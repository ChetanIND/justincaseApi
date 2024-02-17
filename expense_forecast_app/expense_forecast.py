from django.urls import path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

@api_view(['GET'])
@csrf_exempt
def expense_forecast(request):
    # Load the data
    data = pd.read_csv("expense_forecast_app/financedata.csv")

    # Convert the datetime column to datetime format and set it as index
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    data.set_index('Date/Time', inplace=True)

    # Filter data to include only expense values
    expense_data = data[data['Income/Expense'] == 'Expense']

    # Fit the SARIMA model
    sarima_model = SARIMAX(expense_data['Debit/Credit'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit_model = sarima_model.fit()
    # Generate forecasts for the future year (365 days)
    forecast = fit_model.forecast(steps=365)

    # Generate forecasts for the next month (30 days)
    forecast_next_month = fit_model.forecast(steps=30)

    # Plot the original data and forecasted values
    plt.figure(figsize=(20, 6))
    plt.plot(data.index, data['Debit/Credit'], marker='o', linestyle='-', label='Original Data')
    plt.plot(pd.date_range(start=data.index[-1], periods=365, freq='D'), forecast, marker='o', linestyle='--', color='red', label='Forecast')
    plt.title('Expense Forecast')
    plt.xlabel('Date')
    plt.ylabel('Expense')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    imagecount = 0
    image_path = f'data_image//expense_forecast_{imagecount}.png'
    plt.savefig(image_path)
    plt.close() # Close the plot to avoid memory leaks
    imagecount += 1


    # Return the image path and forecast values as JSON response
    response_data = {
        'image_path': image_path,
        'forecast': forecast_next_month.tolist()
    }
    return Response(response_data)