from celery import Celery
from processing import processing
from django.shortcuts import render

app = Celery('tasks', backend='amqp', broker='amqp://')


@app.task
def tskd_processing(request):
    tsk = processing(request)
    return render(request, 'Proceeded.html', {'out': tsk})
