from django.urls import path, include

import recsys.views

urlpatterns = [
    path('hello_world', recsys.views.hello_world),
    path('index', recsys.views.index),
    path('poster_rec', recsys.views.poster_rec),
    path('dilates', recsys.views.dilates),
    path('movie', recsys.views.movie)
]