"""
Matchering WEB - Handy Matchering 2.0 Containerized Web Application
Copyright (C) 2016-2020 Sergree

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = os.environ['SECRET_KEY']

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'rest_framework',
    'django_rq',
    'mgw_back.apps.MgwBackConfig',
    'mgw_front.apps.MgwFrontConfig',
]

ROOT_URLCONF = 'matchering_web.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'APP_DIRS': True,
    },
]

WSGI_APPLICATION = 'matchering_web.wsgi.application'

DB_DIR = os.path.join(BASE_DIR, 'db')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(DB_DIR, 'db.sqlite3'),
    },
}

REST_FRAMEWORK = {
    'UNAUTHENTICATED_USER': None,
}

RQ_QUEUES = {
    'default': {
        'HOST': 'localhost',
        'PORT': 6379,
        'DB': 0,
    },
}

TIME_ZONE = 'UTC'

STATIC_URL = '/static/'

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'
