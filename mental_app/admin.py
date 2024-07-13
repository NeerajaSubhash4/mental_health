from django.contrib import admin
from .models import Question, Registration,Therapist,Booking,Question,Responsemaster,Responsechild,Survey,Questionnaire

admin.site.register(Registration)
admin.site.register(Therapist)
admin.site.register(Booking)
admin.site.register(Question)
admin.site.register(Responsemaster)
admin.site.register(Responsechild)
admin.site.register(Survey)
admin.site.register(Questionnaire)