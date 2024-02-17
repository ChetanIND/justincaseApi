from django.urls import path
from . import views

urlpatterns = [
    path('expense-forecast/', views.expense_forecast),
]