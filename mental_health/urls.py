"""mental_health URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mental_app import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('',views.index),
    path('registration',views.registration),
    path('login',views.login),

    path('adminhome',views.adminhome),
    path('admintherapist',views.admintherapist),
    path('adminpatient',views.adminpatient),

    path('patienthome',views.patienthome),
    path('patientquestion',views.patientquestion),
    path('patientquestionnaire',views.patientquestionnaire),
    path('patientsearchdoctor',views.patientsearchdoctor),
    path('patientbookingdate',views.patientbookingdate),
    path('patientbookings',views.patientbookings),
    path('patientresult',views.patientresult),
    path('patientquestion1',views.patientquestion1),
    path('patientquestionnaire1',views.patientquestionnaire1),
    path('patientbookingdetails',views.patientbookingdetails),

    path('therapisthome',views.therapisthome),
    path('therapistbooking',views.therapistbooking),
    path('therapistpatient',views.therapistpatient),
]
