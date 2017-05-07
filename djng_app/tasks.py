# from celery import Celery, current_task
# from celery.app import task
# from djng_app.processing import processing
# from django.shortcuts import render, redirect
#
# #  celery -A django_proj worker --loglevel=info
#
#
# def tskd_processing(request):
#     try:
#         request.environ['HTTP_REFERER']
#     except:
#         return redirect('http://127.0.0.1:8080')
#     tsk = processing(request)
#     #     current_task.update_state()
#     response = render(request, 'proceeded.html', {'out': tsk})
#
#     return response
