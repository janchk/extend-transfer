from __future__ import absolute_import
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_proj.settings')

app = Celery('django_proj', backend='redis://localhost', broker='pyamqp://guest@localhost//')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print ('Request: {0!r}'.format(self.request))
