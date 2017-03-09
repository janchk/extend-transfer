from celery import Celery
from processing import processing
from django.shortcuts import render,redirect

app = Celery('tasks', backend='amqp', broker='amqp://')


@app.task
def tskd_processing(request):
    try:
        request.environ['HTTP_REFERER']
    except:
        return redirect('http://127.0.0.1:8080')
    tsk = processing(request)
    return render(request, 'Proceeded.html', {'out': tsk})
