from celery import Celery, current_task
from celery.app import task
# from djng_app.style import main
from django_proj.celery import app


@app.task()
def styletransfer(args):
    # processed_output = main(args)
    pass
    # return processed_output



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
